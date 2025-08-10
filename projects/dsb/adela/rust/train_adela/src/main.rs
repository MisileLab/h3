use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use shakmaty::{Chess, Position};
use shakmaty::san::San;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};
use walkdir::WalkDir;

#[derive(Debug, Parser)]
#[command(name = "train_adela")]
#[command(about = "Train Adela MoE model from local Parquet splits")] 
struct Args {
  #[arg(long)]
  data_path: PathBuf,

  #[arg(long, default_value = "models/adela_rust")] 
  output_dir: PathBuf,

  #[arg(long, default_value_t = 50)]
  num_epochs: usize,

  #[arg(long, default_value_t = 256)]
  batch_size: i64,

  #[arg(long, default_value_t = 1000)]
  min_elo: i32,

  #[arg(long, default_value_t = 5)]
  early_stop_patience: usize,

  #[arg(long, default_value_t = 1e-3)]
  early_stop_min_delta: f64,

  #[arg(long, default_value = "auto")]
  device: String,

  /// Use only the first N samples from the train split
  #[arg(long)]
  train_head: Option<usize>,

  /// Use only the first N samples from the validation split
  #[arg(long)]
  val_head: Option<usize>,

  /// Use only the first N samples from the test split
  #[arg(long)]
  test_head: Option<usize>,
}

const POLICY_DIM: i64 = 1968;
const BOARD_CH: i64 = 12;
const BOARD_H: i64 = 8;
const BOARD_W: i64 = 8;
const ADD_FEATS: i64 = 8;

fn resolve_device(arg: &str) -> Device {
  match arg {
    "cpu" => Device::Cpu,
    "cuda" | "gpu" => Device::Cuda(0),
    _ => if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu },
  }
}

fn ensure_splits(root: &Path) -> Result<(PathBuf, PathBuf, PathBuf)> {
  let train = root.join("train");
  let val = root.join("validation");
  let test = root.join("test");
  if train.exists() && val.exists() && test.exists() {
    Ok((train, val, test))
  } else {
    bail!("Expected split folders under {:?} (train/validation/test)", root)
  }
}

fn list_parquet_files(dir: &Path) -> Vec<PathBuf> {
  let mut files = Vec::new();
  for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
    let p = entry.path();
    if p.is_file() && p.extension().map(|e| e == "parquet").unwrap_or(false) {
      files.push(p.to_path_buf());
    }
  }
  files.sort();
  files
}

fn pick_column(df: &DataFrame, candidates: &[&str]) -> Option<String> {
  for c in candidates {
    if df.get_column_names().iter().any(|n| n == c) { return Some((*c).to_string()); }
  }
  let mut lowers = std::collections::HashMap::new();
  for n in df.get_column_names() {
    lowers.insert(n.to_lowercase(), n.to_string());
  }
  for c in candidates {
    if let Some(orig) = lowers.get(&c.to_lowercase()) {
      return Some(orig.clone());
    }
  }
  None
}

fn game_value_from_result(result: &str) -> f32 {
  match result {
    "1-0" => 1.0,
    "0-1" => -1.0,
    _ => 0.0,
  }
}

fn board_tensor_from_pos(board: &Chess) -> Tensor {
  let mut data = vec![0f32; (BOARD_CH * BOARD_H * BOARD_W) as usize];
  for sq in shakmaty::Square::ALL {
    if let Some(piece) = board.board().piece_at(sq) {
      let rank = u8::from(sq.rank()) as i64;
      let file = u8::from(sq.file()) as i64;
      let channel = match (piece.color, piece.role) {
        (shakmaty::Color::White, shakmaty::Role::Pawn) => 0,
        (shakmaty::Color::White, shakmaty::Role::Knight) => 1,
        (shakmaty::Color::White, shakmaty::Role::Bishop) => 2,
        (shakmaty::Color::White, shakmaty::Role::Rook) => 3,
        (shakmaty::Color::White, shakmaty::Role::Queen) => 4,
        (shakmaty::Color::White, shakmaty::Role::King) => 5,
        (shakmaty::Color::Black, shakmaty::Role::Pawn) => 6,
        (shakmaty::Color::Black, shakmaty::Role::Knight) => 7,
        (shakmaty::Color::Black, shakmaty::Role::Bishop) => 8,
        (shakmaty::Color::Black, shakmaty::Role::Rook) => 9,
        (shakmaty::Color::Black, shakmaty::Role::Queen) => 10,
        (shakmaty::Color::Black, shakmaty::Role::King) => 11,
      } as i64;
      let offset = (channel * BOARD_H * BOARD_W + rank * BOARD_W + file) as usize;
      data[offset] = 1.0;
    }
  }
  Tensor::from_slice(&data).reshape([BOARD_CH, BOARD_H, BOARD_W])
}

fn additional_features_from_pos(board: &Chess) -> Tensor {
  let mut f = [0f32; 8];
  f[0] = if board.turn() == shakmaty::Color::White { 1.0 } else { 0.0 };
  f[1] = 0.0; f[2] = 0.0; f[3] = 0.0; f[4] = 0.0; // castling placeholders
  f[5] = 0.0; // en-passant placeholder
  f[6] = (board.halfmoves() as f32 / 100.0).min(1.0);
  let mut total_material = 0i32;
  use shakmaty::Role;
  for role in [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen] {
    for sq in shakmaty::Square::ALL {
      if let Some(p) = board.board().piece_at(sq) {
        if p.role == role { total_material += 1; }
      }
    }
  }
  let phase = 1.0 - (total_material as f32 / 30.0);
  f[7] = phase.max(0.0).min(1.0);
  Tensor::from_slice(&f)
}

fn process_moves_into_samples(moves_san: Vec<String>, result_opt: Option<String>) -> Vec<(Tensor, Tensor, Tensor, f32)> {
  let mut samples = Vec::new();
  let mut board = Chess::default();
  let game_value = game_value_from_result(result_opt.as_deref().unwrap_or("*"));
  let limit = moves_san.len().min(30);
  for san in moves_san.into_iter().take(limit) {
    let board_tensor = board_tensor_from_pos(&board);
    let add_feats = additional_features_from_pos(&board);
    let policy = Tensor::zeros([POLICY_DIM], (Kind::Float, Device::Cpu));
    let value = if board.turn() == shakmaty::Color::White { game_value } else { -game_value };
    samples.push((board_tensor, add_feats, policy, value));

    if let Ok(m) = san.parse::<San>() {
      if let Ok(mv) = m.to_move(&board) { board.play_unchecked(mv); } else { break; }
    } else { break; }
  }
  samples
}

fn load_split_as_iter(root: &Path, limit: Option<usize>) -> Result<Vec<(Tensor, Tensor, Tensor, f32)>> {
  let files = list_parquet_files(root);
  if files.is_empty() { return Ok(vec![]); }
  let mut all = Vec::new();
  let mut total_samples: usize = 0;
  'files: for (i, fpath) in files.iter().enumerate() {
    let file = std::fs::File::open(fpath).with_context(|| format!("open parquet {:?}", fpath))?;
    let df = ParquetReader::new(file).finish().with_context(|| format!("read parquet {:?}", fpath))?;

    let parsed_moves_col = match pick_column(&df, &["parsed_moves"]) { Some(c) => c, None => continue };
    let result_col = pick_column(&df, &["result", "Result"]);

    let moves_sc = df.column(&parsed_moves_col)?.str()?;
    let result_sc_opt: Option<StringChunked> = match result_col {
      Some(c) => Some(df.column(&c)?.str()?.clone()),
      None => None,
    };

    let n = moves_sc.len();
    let pb = if let Some(lim) = limit { ProgressBar::new(lim as u64) } else { ProgressBar::new(n as u64) };
    pb.set_style(ProgressStyle::with_template("{msg} {bar:40.cyan/blue} {pos}/{len}").unwrap());
    pb.set_message(format!("{:>3}/{:>3} processing", i + 1, files.len()));

    for idx in 0..n {
      if let Some(lim) = limit { if total_samples >= lim { pb.finish_and_clear(); break 'files; } }

      let moves_san: Vec<String> = match moves_sc.get(idx) {
        Some(s) => s.split_whitespace().map(|m| m.to_string()).collect(),
        None => { if limit.is_none() { pb.inc(1); } continue; }
      };
      if moves_san.is_empty() { if limit.is_none() { pb.inc(1); } continue; }
      let result_opt = result_sc_opt.as_ref().and_then(|sc| sc.get(idx)).map(|s| s.to_string());
      let samples = process_moves_into_samples(moves_san, result_opt);
      if let Some(lim) = limit {
        let remaining = lim.saturating_sub(total_samples);
        let to_take = samples.len().min(remaining);
        for (bt, af, pol, val) in samples.into_iter().take(to_take) { all.push((bt, af, pol, val)); }
        total_samples += to_take;
        pb.inc(to_take as u64);
      } else {
        for (bt, af, pol, val) in samples { all.push((bt, af, pol, val)); }
        pb.inc(1);
      }
    }
    pb.finish_and_clear();
  }
  Ok(all)
}

struct MoEModel {
  vs: nn::VarStore,
  conv1: nn::Conv2D,
  bn1: nn::BatchNorm,
  policy_head: nn::Linear,
  value_head1: nn::Linear,
  value_head2: nn::Linear,
}

impl MoEModel {
  fn new(device: Device) -> Self {
    let vs = nn::VarStore::new(device);
    let root = &vs.root();
    let conv1 = nn::conv2d(root, BOARD_CH, 256, 3, nn::ConvConfig { padding: 1, ..Default::default() });
    let bn1 = nn::batch_norm2d(root, 256, Default::default());
    // 256 channels * 8 * 8 spatial size
    let policy_head = nn::linear(root, 256 * 8 * 8, POLICY_DIM, Default::default());
    // concat([256*8*8, 32]) => 16384 + 32
    let value_head1 = nn::linear(root, 256 * 8 * 8 + 32, 256, Default::default());
    let value_head2 = nn::linear(root, 256, 1, Default::default());
    Self { vs, conv1, bn1, policy_head, value_head1, value_head2 }
  }

  fn forward(&self, boards: &Tensor, feats: &Tensor, train: bool) -> (Tensor, Tensor) {
    let x = boards.apply(&self.conv1).apply_t(&self.bn1, train).relu();
    let flat = x.avg_pool2d_default(1).flatten(1, -1);
    let policy = flat.apply(&self.policy_head);
    let feats_proj_w = nn::linear(&self.vs.root(), ADD_FEATS, 32, Default::default());
    let feats_proj = feats.apply(&feats_proj_w);
    let vcat = Tensor::cat(&[flat, feats_proj], 1);
    let value = vcat.apply(&self.value_head1).relu().apply(&self.value_head2).tanh();
    (policy, value)
  }
}

fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor { (pred - target).pow_tensor_scalar(2).mean(Kind::Float) }

fn tensor_to_f64_scalar(x: &Tensor) -> f64 { x.to_device(Device::Cpu).double_value(&[]) }

fn train_loop(args: &Args) -> Result<()> {
  fs::create_dir_all(&args.output_dir)?;
  let device = resolve_device(&args.device);

  let (train_dir, val_dir, test_dir) = ensure_splits(&args.data_path)
    .or_else(|_| ensure_splits(&args.data_path.join("data")))?;

  println!("Loading splits...");
  let train_samples = load_split_as_iter(&train_dir, args.train_head)?;
  let val_samples = load_split_as_iter(&val_dir, args.val_head)?;
  let test_samples = load_split_as_iter(&test_dir, args.test_head)?;

  if train_samples.is_empty() || val_samples.is_empty() || test_samples.is_empty() {
    bail!("One of the splits is empty after processing");
  }
  println!(
    "Train: {}  Val: {}  Test: {} positions",
    train_samples.len(),
    val_samples.len(),
    test_samples.len()
  );

  let model = MoEModel::new(device);
  let mut opt = nn::Adam::default().build(&model.vs, 1e-3)?;

  let mut best_val = f64::INFINITY;
  let mut no_improve = 0usize;

  for epoch in 0..args.num_epochs {
    println!("Epoch {}/{}", epoch + 1, args.num_epochs);

    let mut total_loss = 0f64;
    let mut count = 0usize;

    for batch in train_samples.chunks(args.batch_size as usize) {
      let boards = Tensor::stack(&batch.iter().map(|(b, _, _, _)| b.shallow_clone()).collect::<Vec<_>>(), 0).to_device(device).to_kind(Kind::Float);
      let feats = Tensor::stack(&batch.iter().map(|(_, f, _, _)| f.shallow_clone()).collect::<Vec<_>>(), 0).to_device(device).to_kind(Kind::Float);
      let policy_target = Tensor::stack(&batch.iter().map(|(_, _, p, _)| p.shallow_clone()).collect::<Vec<_>>(), 0).to_device(device).to_kind(Kind::Float);
      let value_target = Tensor::from_slice(&batch.iter().map(|(_, _, _, v)| *v).collect::<Vec<_>>()).to_device(device).reshape([-1, 1]).to_kind(Kind::Float);

      let (policy_logits, value_pred) = model.forward(&boards, &feats, true);
      let target_idx = policy_target.argmax(Some(1), false).to_kind(Kind::Int64);
      let policy_loss = policy_logits.cross_entropy_for_logits(&target_idx);
      let value_loss = mse_loss(&value_pred.squeeze_dim(-1), &value_target.squeeze_dim(-1));
      let loss = &policy_loss + &value_loss;

      opt.backward_step(&loss);
      total_loss += tensor_to_f64_scalar(&loss);
      count += 1;
    }
    let train_avg = if count > 0 { total_loss / count as f64 } else { 0.0 };
    println!("train loss: {:.4}", train_avg);

    let mut val_total = 0f64;
    let mut val_count = 0usize;
    for batch in val_samples.chunks(args.batch_size as usize) {
      let boards = Tensor::stack(&batch.iter().map(|(b, _, _, _)| b.shallow_clone()).collect::<Vec<_>>(), 0).to_device(device).to_kind(Kind::Float);
      let feats = Tensor::stack(&batch.iter().map(|(_, f, _, _)| f.shallow_clone()).collect::<Vec<_>>(), 0).to_device(device).to_kind(Kind::Float);
      let policy_target = Tensor::stack(&batch.iter().map(|(_, _, p, _)| p.shallow_clone()).collect::<Vec<_>>(), 0).to_device(device).to_kind(Kind::Float);
      let value_target = Tensor::from_slice(&batch.iter().map(|(_, _, _, v)| *v).collect::<Vec<_>>()).to_device(device).reshape([-1, 1]).to_kind(Kind::Float);

      let (policy_logits, value_pred) = model.forward(&boards, &feats, false);
      let target_idx = policy_target.argmax(Some(1), false).to_kind(Kind::Int64);
      let policy_loss = policy_logits.cross_entropy_for_logits(&target_idx);
      let value_loss = mse_loss(&value_pred.squeeze_dim(-1), &value_target.squeeze_dim(-1));
      let loss = &policy_loss + &value_loss;
      val_total += tensor_to_f64_scalar(&loss);
      val_count += 1;
    }
    let val_avg = if val_count > 0 { val_total / val_count as f64 } else { f64::INFINITY };
    println!("val loss: {:.4}", val_avg);

    if best_val - val_avg > args.early_stop_min_delta {
      best_val = val_avg;
      no_improve = 0;
      let ckpt = args.output_dir.join("model_best.safetensors");
      model.vs.save(ckpt)?;
      println!("improved -> saved best checkpoint");
    } else {
      no_improve += 1;
      println!("no improvement ({}/{})", no_improve, args.early_stop_patience);
      if no_improve >= args.early_stop_patience { break; }
    }
  }

  let final_ckpt = args.output_dir.join("model_final.safetensors");
  model.vs.save(final_ckpt)?;
  println!("saved final model");

  let mut test_total = 0f64;
  let mut test_count = 0usize;
  for batch in test_samples.chunks(args.batch_size as usize) {
    let boards = Tensor::stack(&batch.iter().map(|(b, _, _, _)| b.shallow_clone()).collect::<Vec<_>>(), 0).to_device(device).to_kind(Kind::Float);
    let feats = Tensor::stack(&batch.iter().map(|(_, f, _, _)| f.shallow_clone()).collect::<Vec<_>>(), 0).to_device(device).to_kind(Kind::Float);
    let policy_target = Tensor::stack(&batch.iter().map(|(_, _, p, _)| p.shallow_clone()).collect::<Vec<_>>(), 0).to_device(device).to_kind(Kind::Float);
    let value_target = Tensor::from_slice(&batch.iter().map(|(_, _, _, v)| *v).collect::<Vec<_>>()).to_device(device).reshape([-1, 1]).to_kind(Kind::Float);

    let (policy_logits, value_pred) = model.forward(&boards, &feats, false);
    let target_idx = policy_target.argmax(Some(1), false).to_kind(Kind::Int64);
    let policy_loss = policy_logits.cross_entropy_for_logits(&target_idx);
    let value_loss = mse_loss(&value_pred.squeeze_dim(-1), &value_target.squeeze_dim(-1));
    let loss = &policy_loss + &value_loss;
    test_total += tensor_to_f64_scalar(&loss);
    test_count += 1;
  }
  let test_avg = if test_count > 0 { test_total / test_count as f64 } else { f64::INFINITY };
  println!("test loss: {:.4}", test_avg);

  Ok(())
}

fn main() -> Result<()> {
  let args = Args::parse();
  train_loop(&args)
}
