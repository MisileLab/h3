use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn};
use once_cell::sync::Lazy;
use polars::prelude::*;
use regex::Regex;

static RE_RESULT_END: Lazy<Regex> = Lazy::new(|| Regex::new(r"(1-0|0-1|1/2-1/2|\*)\s*$").unwrap());
static RE_WHITESPACE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());
static RE_SPLIT_MOVENUM: Lazy<Regex> = Lazy::new(|| Regex::new(r"\d+\.").unwrap());
static RE_SAN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(?:[KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBNP])?[+#]?|O-O(?:-O)?[+#]?)\b").unwrap()
});

fn parse_movetext_to_moves(movetext: &str) -> Vec<String> {
    if movetext.trim().is_empty() {
        return Vec::new();
    }
    let cleaned = RE_RESULT_END.replace(movetext, "");
    let cleaned = RE_WHITESPACE.replace_all(cleaned.trim(), " ");

    let mut moves: Vec<String> = Vec::new();
    for token in RE_SPLIT_MOVENUM.split(&cleaned) {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        for m in RE_SAN.find_iter(token) {
            let mv = m.as_str();
            if !(mv == "1-0" || mv == "0-1" || mv == "1/2-1/2" || mv == "*")
                && mv.len() >= 2
                && !mv.chars().all(|c| c.is_ascii_digit())
            {
                moves.push(mv.to_string());
            }
        }
    }
    moves
}

#[derive(Parser, Debug)]
#[command(name = "load_and_parse_parquets")]
#[command(about = "Load Parquet files, filter by Elo, and parse movetext", long_about = None)]
struct Cli {
    folder_path: PathBuf,
    #[arg(long, default_value_t = 2000)]
    min_elo: i64,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    output_dir: Option<PathBuf>,
    #[arg(long, default_value = "processed_games")]
    output_prefix: String,
    #[arg(long, default_value_t = 5_000)]
    batch_size: usize,
    #[arg(long, default_value_t = false)]
    batch_mode: bool,
    #[arg(long, default_value_t = 5_000)]
    file_chunk_size: usize,
    #[arg(long, default_value_t = 500)]
    parse_chunk_size: usize,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long)]
    sample: Option<usize>,
    #[arg(long, default_value_t = false)]
    show_examples: bool,
}

fn ensure_dir(path: &Path) -> Result<()> {
    if !path.exists() {
        fs::create_dir_all(path).with_context(|| format!("failed to create dir {path:?}"))?;
    }
    Ok(())
}

fn list_parquet_files(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.exists() {
        return Err(anyhow!("Folder does not exist: {:?}", dir));
    }
    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "parquet").unwrap_or(false))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(anyhow!("No Parquet files found in folder: {:?}", dir));
    }
    Ok(files)
}

fn find_column_case_insensitive(df: &DataFrame, candidates: &[&str]) -> Option<String> {
    let cols: Vec<&PlSmallStr> = df.get_column_names();
    for cand in candidates {
        for c in &cols {
            if *c == *cand {
                return Some((*c).to_string());
            }
            if c.eq_ignore_ascii_case(cand) {
                return Some((*c).to_string());
            }
        }
    }
    None
}

fn get_total_rows_parquet(path: &Path) -> Result<i64> {
    let lf = LazyFrame::scan_parquet(PlPath::new(path.to_string_lossy().as_ref()), Default::default())?;
    let out = lf.select([len()]).collect()?;
    let s = out.get_columns()[0].clone();
    let total = s.get(0).unwrap().try_extract::<i64>().unwrap_or(0);
    Ok(total)
}

fn coerce_to_i64(df: &mut DataFrame, col: &str) -> Result<()> {
    if df.column(col)?.dtype() != &DataType::Int64 {
        let s = df.column(col)?.cast(&DataType::Int64)?.as_series().unwrap().clone();
        df.replace(col, s)?;
    }
    Ok(())
}

fn read_slice_and_filter_elo(path: &Path, start: i64, len_rows: i64, min_elo: i64) -> Result<DataFrame> {
    let lf = LazyFrame::scan_parquet(PlPath::new(path.to_string_lossy().as_ref()), Default::default())?
        .slice(start, len_rows as IdxSize);
    let mut df = lf.collect()?;

    let white_col = find_column_case_insensitive(&df, &["WhiteElo", "white_elo"]).ok_or_else(|| anyhow!("No WhiteElo/white_elo column"))?;
    let black_col = find_column_case_insensitive(&df, &["BlackElo", "black_elo"]).ok_or_else(|| anyhow!("No BlackElo/black_elo column"))?;

    coerce_to_i64(&mut df, &white_col)?;
    coerce_to_i64(&mut df, &black_col)?;

    let left = df.column(&white_col)?.i64()?.gt(min_elo - 1);
    let right = df.column(&black_col)?.i64()?.gt(min_elo - 1);
    let classical = df.column("Event")?.str()?.equal("Rated Classical Game");
    let and_mask = (&left & &right) & classical;

    let filtered = df.filter(&and_mask)?;
    Ok(filtered)
}

fn vstack_all(mut dfs: Vec<DataFrame>) -> Result<DataFrame> {
    if dfs.is_empty() {
        return Ok(DataFrame::empty());
    }
    let mut acc = dfs.remove(0);
    for df in dfs.into_iter() {
        acc.vstack_mut(&df)?;
    }
    Ok(acc)
}

fn add_parsed_moves(df: &DataFrame, chunk_size: usize) -> Result<DataFrame> {
    let movetext_col = if df.get_column_names().iter().any(|c| c == &"movetext") {
        "movetext".to_string()
    } else if df.get_column_names().iter().any(|c| c == &"moves") {
        "moves".to_string()
    } else {
        let names = df.get_column_names();
        let lower = names
            .iter()
            .find(|c| c.to_lowercase().contains("move"))
            .ok_or_else(|| anyhow!("No movetext/moves column found"))?;
        warn!("Using column '{}' as movetext source", lower);
        (*lower).to_string()
    };

    let movetext = df.column(&movetext_col)?.str()?;
    let total_rows = movetext.len();

    let mut parsed_strs: Vec<Option<String>> = Vec::with_capacity(total_rows);
    let mut counts: Vec<u32> = Vec::with_capacity(total_rows);

    for (idx, opt_val) in movetext.into_iter().enumerate() {
        if total_rows > 5000 && idx % chunk_size == 0 {
            // pacing
        }
        let moves_vec: Vec<String> = match opt_val {
            Some(s) => parse_movetext_to_moves(s),
            None => Vec::new(),
        };
        counts.push(moves_vec.len() as u32);
        parsed_strs.push(Some(moves_vec.join(" ")));
    }

    let parsed_moves_series = Series::new("parsed_moves".into(), parsed_strs);
    let num_moves_series = Series::new("num_moves".into(), counts);

    let mut out = df.clone();
    out.with_column(parsed_moves_series)?;
    out.with_column(num_moves_series)?;
    out = out.drop_many(vec!["Event", "Site", "White", "Black", "WhiteTitle", "BlackTitle", "WhiteRatingDiff", "BlackRatingDiff", "UTCDate", "UTCTime", "ECO", "Opening", "Termination", "TimeControl", "movetext"]);
    Ok(out)
}

fn save_parquet(df: &DataFrame, output_file: &Path) -> Result<()> {
    if let Some(dir) = output_file.parent() {
        ensure_dir(dir)?;
    }
    let file = File::create(output_file).with_context(|| format!("cannot create file {:?}", output_file))?;
    let mut writer = ParquetWriter::new(file);
    writer = writer.with_compression(ParquetCompression::Zstd(None));
    let mut df_owned = df.clone();
    writer.finish(&mut df_owned)?;
    Ok(())
}

fn save_batch_df(batch_df: &DataFrame, batch_num: usize, output_dir: &Path, prefix: &str) -> Result<PathBuf> {
    if batch_df.height() == 0 {
        return Err(anyhow!("empty batch"));
    }
    let filename = format!("{}_batch_{:06}.parquet", prefix, batch_num);
    let path = output_dir.join(filename);
    save_parquet(batch_df, &path)?;
    Ok(path)
}

fn process_parquet_files_in_batches(
    folder_path: &Path,
    min_elo: i64,
    batch_size: usize,
    output_dir: &Path,
    output_prefix: &str,
    file_chunk_size: usize,
    parse_chunk_size: usize,
) -> Result<Vec<PathBuf>> {
    let parquet_files = list_parquet_files(folder_path)?;

    info!("Found {} Parquet files in {:?}", parquet_files.len(), folder_path);
    info!("Processing in batches of {} games", batch_size);

    ensure_dir(output_dir)?;

    // Precompute total rows
    let mut grand_total_rows: i64 = 0;
    let mut file_to_total: Vec<(PathBuf, i64)> = Vec::with_capacity(parquet_files.len());
    for f in &parquet_files {
        match get_total_rows_parquet(f) {
            Ok(n) => {
                file_to_total.push((f.clone(), n));
                grand_total_rows += n;
            }
            Err(_) => {
                file_to_total.push((f.clone(), 0));
            }
        }
    }

    let disable_games_bar = grand_total_rows <= 0;
    let pb = if disable_games_bar {
        ProgressBar::hidden()
    } else {
        ProgressBar::new(grand_total_rows as u64)
    };
    if !disable_games_bar {
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} {bar:40.cyan/blue} {pos}/{len} rows | included {msg}",
            )
            .unwrap(),
        );
    }

    let mut current_batch: Vec<DataFrame> = Vec::new();
    let mut batch_num: usize = 0;
    let mut output_files: Vec<PathBuf> = Vec::new();
    let mut total_included: i64 = 0;

    let files_pb = ProgressBar::new(parquet_files.len() as u64);
    files_pb.set_style(ProgressStyle::with_template("{spinner:.green} files {pos}/{len}").unwrap());

    for (idx, (file_path, total_rows)) in file_to_total.into_iter().enumerate() {
        info!(
            "Processing file {}/{}: {}",
            idx + 1,
            parquet_files.len(),
            file_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
        );
        let mut chunk_start: i64 = 0;
        while chunk_start < total_rows {
            let rows_in_slice = std::cmp::min(file_chunk_size as i64, total_rows - chunk_start);
            let df = read_slice_and_filter_elo(&file_path, chunk_start, rows_in_slice, min_elo)?;
            if df.height() == 0 {
                if !disable_games_bar {
                    pb.inc(rows_in_slice as u64);
                    pb.set_message(format!("{}", total_included));
                }
                chunk_start += rows_in_slice;
                continue;
            }
            let parsed = add_parsed_moves(&df, parse_chunk_size)?;
            total_included += parsed.height() as i64;
            current_batch.push(parsed);

            let mut combined = vstack_all(current_batch)?;
            while combined.height() >= batch_size {
                let batch_df = combined.slice(0, batch_size);
                let out_file = save_batch_df(&batch_df, batch_num, output_dir, output_prefix)?;
                output_files.push(out_file);
                info!(
                    "Saved batch {} with {} games",
                    batch_num,
                    batch_df.height()
                );
                batch_num += 1;
                combined = combined.slice(batch_size as i64, combined.height() - batch_size);
            }
            current_batch = if combined.height() > 0 {
                vec![combined]
            } else {
                Vec::new()
            };

            if !disable_games_bar {
                pb.inc(rows_in_slice as u64);
                pb.set_message(format!("{}", total_included));
            }
            chunk_start += rows_in_slice;
        }
        files_pb.inc(1);
    }

    // Finalize remaining
    if !current_batch.is_empty() {
        let mut combined = vstack_all(current_batch)?;
        while combined.height() >= batch_size {
            let batch_df = combined.slice(0, batch_size);
            let out_file = save_batch_df(&batch_df, batch_num, output_dir, output_prefix)?;
            output_files.push(out_file);
            info!(
                "Saved batch {} with {} games",
                batch_num,
                batch_df.height()
            );
            batch_num += 1;
            combined = combined.slice(batch_size as i64, combined.height() - batch_size);
        }
        if combined.height() > 0 {
            let out_file = save_batch_df(&combined, batch_num, output_dir, output_prefix)?;
            output_files.push(out_file);
            info!(
                "Saved final batch {} with {} games",
                batch_num,
                combined.height()
            );
        }
    }

    if !disable_games_bar {
        pb.finish_and_clear();
    }
    files_pb.finish_and_clear();

    info!("Processing complete: included {} games", total_included);
    info!("Created {} batch files in {:?}", output_files.len(), output_dir);

    Ok(output_files)
}

fn load_parquet_files(folder_path: &Path, max_memory_mb: usize) -> Result<DataFrame> {
    let files = list_parquet_files(folder_path)?;
    info!("Found {} Parquet files in {:?}", files.len(), folder_path);

    let total_size_mb: f64 = files
        .iter()
        .filter_map(|f| fs::metadata(f).ok())
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .sum();
    if total_size_mb > max_memory_mb as f64 {
        warn!(
            "Dataset size ({:.1}MB) exceeds recommended limit ({}MB). Consider --batch-mode.",
            total_size_mb, max_memory_mb
        );
    }

    if files.len() == 1 {
        panic!("Only one Parquet file found. Please provide a folder with multiple Parquet files.");
    } else {
        let avg_file_size_mb = (total_size_mb / files.len() as f64).max(1.0);
        let max_files_per_group = (max_memory_mb as f64 / avg_file_size_mb)
            .floor()
            .clamp(1.0, 10.0) as usize;
        let mut all: Vec<DataFrame> = Vec::new();
        for group in files.chunks(max_files_per_group) {
            let mut group_dfs: Vec<DataFrame> = Vec::with_capacity(group.len());
            for file in group {
                let df = LazyFrame::scan_parquet(PlPath::new(file.to_string_lossy().as_ref()), Default::default())?
                    .collect()?;
                group_dfs.push(df);
            }
            let df_group = vstack_all(group_dfs)?;
            all.push(df_group);
        }
        let df_all = vstack_all(all)?;
        Ok(df_all)
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    if cli.batch_mode {
        info!("Using batch processing mode for large dataset");
        let out_dir = cli
            .output_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("."));
        let outputs = process_parquet_files_in_batches(
            &cli.folder_path,
            cli.min_elo,
            cli.batch_size,
            &out_dir,
            &cli.output_prefix,
            cli.file_chunk_size,
            cli.parse_chunk_size,
        )?;
        println!("\nBatch processing complete!");
        println!("Created {} batch files in {:?}", outputs.len(), out_dir);
        if !outputs.is_empty() {
            let show: Vec<String> = outputs
                .iter()
                .take(5)
                .map(|p| p.file_name().unwrap_or_default().to_string_lossy().to_string())
                .collect();
            println!("Batch files: {:?}...", show);
        }
    } else {
        info!("Using single file processing mode");
        warn!("Note: For datasets >5GB, consider using --batch-mode for better memory efficiency");

        let mut df = load_parquet_files(&cli.folder_path, 2048)?;

        if let Some(n) = cli.sample {
            df = df.head(Some(n));
            info!("Sampled {} games for processing", n);
        }

        let white_col = find_column_case_insensitive(&df, &["WhiteElo", "white_elo"]).ok_or_else(|| anyhow!("No WhiteElo/white_elo column"))?;
        let black_col = find_column_case_insensitive(&df, &["BlackElo", "black_elo"]).ok_or_else(|| anyhow!("No BlackElo/black_elo column"))?;

        {
            let mut tmp = df.clone();
            coerce_to_i64(&mut tmp, &white_col)?;
            coerce_to_i64(&mut tmp, &black_col)?;
            df = tmp;
        }

        let left = df.column(&white_col)?.i64()?.gt(cli.min_elo - 1);
        let right = df.column(&black_col)?.i64()?.gt(cli.min_elo - 1);
        let or_mask = &left | &right;
        df = df.filter(&or_mask)?;

        if df.height() == 0 {
            warn!("No games remaining after Elo filtering");
            return Ok(());
        }

        let chunk_size = if df.height() > 1000 {
            cli.parse_chunk_size
        } else {
            cli.parse_chunk_size.min(100)
        };
        df = add_parsed_moves(&df, chunk_size)?;

        info!("Final dataset: {} games", df.height());

        let num_moves = df.column("num_moves")?.u32()?;
        let mut min_v = u32::MAX;
        let mut max_v = 0u32;
        let mut sum_v: u64 = 0;
        let mut vals: Vec<u32> = Vec::with_capacity(num_moves.len());
        for v in num_moves.into_no_null_iter() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
            sum_v += v as u64;
            vals.push(v);
        }
        vals.sort_unstable();
        let median_v = if vals.is_empty() { 0 } else { vals[vals.len() / 2] };
        let avg_v = if !vals.is_empty() {
            sum_v as f64 / vals.len() as f64
        } else {
            0.0
        };
        info!(
            "Move statistics: avg={:.1}, min={}, max={}, median={}",
            avg_v, min_v, max_v, median_v
        );

        if cli.show_examples {
            println!("\n=== Example Parsed Moves ===");
            let w_name = find_column_case_insensitive(&df, &["white", "White"]).unwrap_or_else(|| "white".to_string());
            let b_name = find_column_case_insensitive(&df, &["black", "Black"]).unwrap_or_else(|| "black".to_string());
            let take = df.head(Some(3));
            for i in 0..take.height() {
                let get_str = |df: &DataFrame, name: &str| -> Option<String> {
                    df.column(name).ok()?.str().ok()?.get(i).map(|s| s.to_string())
                };
                let white = get_str(&take, &w_name).unwrap_or_default();
                let black = get_str(&take, &b_name).unwrap_or_default();
                let w_elo = take
                    .column(&white_col)
                    .ok()
                    .and_then(|s| s.get(i).ok())
                    .map(|v| format!("{}", v))
                    .unwrap_or_default();
                let b_elo = take
                    .column(&black_col)
                    .ok()
                    .and_then(|s| s.get(i).ok())
                    .map(|v| format!("{}", v))
                    .unwrap_or_default();
                let parsed_str = take
                    .column("parsed_moves")
                    .unwrap()
                    .str()
                    .unwrap()
                    .get(i)
                    .unwrap_or("")
                    .to_string();
                let num_moves = take
                    .column("num_moves")
                    .unwrap()
                    .u32()
                    .unwrap()
                    .get(i)
                    .unwrap_or(0);
                let preview: Vec<&str> = parsed_str.split_whitespace().take(10).collect();
                println!(
                    "\nGame {}: {} ({}) vs {} ({})",
                    i + 1,
                    white,
                    w_elo,
                    black,
                    b_elo
                );
                println!("Moves ({}): {:?}...", num_moves, preview);
            }
        }

        if let Some(out) = cli.output {
            save_parquet(&df, &out)?;
            info!("Saved processed data to {:?}", out);
        }

        println!("\nProcessing complete! {} games processed.", df.height());
    }

    Ok(())
}
