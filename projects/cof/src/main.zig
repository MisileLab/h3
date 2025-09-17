const std = @import("std");

// Helper to convert bytes to a hex string.
fn bytesToHex(bytes: []const u8, out: []u8) void {
    const hex_chars = "0123456789abcdef";
    for (bytes, 0..) |b, i| {
        out[i * 2] = hex_chars[b >> 4];
        out[i * 2 + 1] = hex_chars[b & 0x0F];
    }
}

// Processes a single file: chunks, hashes, and stores it in the object store.
fn processPath(allocator: std.mem.Allocator, path: []const u8) !std.ArrayList([32]u8) {
    var block_hashes = std.ArrayList([32]u8).init(allocator);
    errdefer block_hashes.deinit();

    const block_size = 4096;
    var buffer: [block_size]u8 = undefined;
    var hex_buffer: [64]u8 = undefined;

    const cwd = std.fs.cwd();
    var objects_dir = try cwd.openDir(".cof/objects/hot", .{{}});
    defer objects_dir.close();

    var file = try cwd.openFile(path, .{{}});
    defer file.close();

    while (true) {
        const bytes_read = try file.read(&buffer);
        if (bytes_read == 0) break;
        const chunk = buffer[0..bytes_read];

        var hasher = std.crypto.hash.Blake3.init(.{{}});
        hasher.update(chunk);
        var hash: [32]u8 = undefined;
        hasher.final(&hash);

        try block_hashes.append(hash);

        bytesToHex(&hash, &hex_buffer);
        const hash_hex = hex_buffer[0..];

        const stat = objects_dir.statFile(hash_hex) catch |err| {{
            if (err == error.FileNotFound) {{
                var block_file = try objects_dir.createFile(hash_hex, .{{}});
                defer block_file.close();
                try block_file.writeAll(chunk);
            }} else {{
                return err;
            }}
            continue;
        }};
        _ = stat;
    }
    return block_hashes;
}

fn getFileList(allocator: std.mem.Allocator, file_buffer: *[1024][]const u8, count: *usize) !void {
    // 1. Load ignore patterns
    var ignore_patterns: [128][]const u8 = undefined;
    var ignore_count: usize = 0;
    const ignore_file = std.fs.cwd().openFile(".cofignore", .{{}}) catch |err| switch (err) {{
        error.FileNotFound => null,
        else => return err,
    }};

    if (ignore_file) |file| {{
        defer file.close();
        const stat = try file.stat();
        const content = try allocator.alloc(u8, stat.size);
        defer allocator.free(content);
        _ = try file.readAll(content);

        var it = std.mem.splitSequence(u8, content, "\n");
        while (it.next()) |line| {{
            if (ignore_count >= 128) break;
            const trimmed = std.mem.trim(u8, line, " \r");
            if (trimmed.len == 0) continue;
            ignore_patterns[ignore_count] = try allocator.dupe(u8, trimmed);
            ignore_count += 1;
        }}
    }}
    defer {{
        for (ignore_patterns[0..ignore_count]) |p| allocator.free(p);
    }}

    // 2. Walk directory
    var dir = try std.fs.cwd().openDir(".", .{{ .iterate = true }});
    defer dir.close();
    var walker = try dir.walk(allocator);
    defer walker.deinit();

    outer: while (try walker.next()) |entry| {{
        if (std.mem.startsWith(u8, entry.path, ".cof")) {{
            continue :outer;
        }}

        for (ignore_patterns[0..ignore_count]) |pattern| {{
            if (std.mem.endsWith(u8, pattern, "/")) {{
                const dir_name = pattern[0 .. pattern.len - 1];
                if (std.mem.startsWith(u8, entry.path, dir_name)) {{
                    continue :outer;
                }}
            }} else if (std.mem.startsWith(u8, pattern, "*.")) {{
                if (entry.kind == .file and std.mem.endsWith(u8, entry.path, pattern[1..])) {{
                    continue :outer;
                }}
            }}
        }}

        if (entry.kind == .file) {{
            if (count.* >= 1024) break;
            if (std.mem.eql(u8, entry.path, ".cofignore")) continue;
            if (std.mem.eql(u8, entry.path, ".gitignore")) continue;

            file_buffer[count.*] = try allocator.dupe(u8, entry.path);
            count.* += 1;
        }}
    }}
}

// Executes the commit process.
fn commit(allocator: std.mem.Allocator, message: []const u8) !void {{
    _ = message;

    var file_buffer: [1024][]const u8 = undefined;
    var file_count: usize = 0;
    try getFileList(allocator, &file_buffer, &file_count);

    const file_list = file_buffer[0..file_count];

    defer {{
        for (file_list) |item| {{
            allocator.free(item);
        }}
    }}

    for (file_list) |file_path| {{
        std.debug.print("Processing: {s}\n", .{{file_path}});
        var hashes = try processPath(allocator, file_path);
        defer hashes.deinit();
        std.debug.print("  Blocks: {d}\n", .{{hashes.items.len}});
    }}

    std.debug.print("Commit successful (object generation only).\n", .{{}});
}}

// Initializes a new .cof repository.
fn initRepository() !void {{
    const fs = std.fs;
    const cwd = fs.cwd();

    try cwd.makeDir(".cof");
    var cof_dir = try cwd.openDir(".cof", .{{}});
    defer cof_dir.close();

    try cof_dir.makeDir("objects");
    var objects_dir = try cof_dir.openDir("objects", .{{}});
    defer objects_dir.close();
    try objects_dir.makeDir("hot");
    try objects_dir.makeDir("warm");
    try objects_dir.makeDir("cold");

    try cof_dir.makeDir("index");
    try cof_dir.makeDir("refs");
    var refs_dir = try cof_dir.openDir("refs", .{{}});
    defer refs_dir.close();
    try refs_dir.makeDir("heads");

    try cof_dir.makeDir("locks");

    var config_file = try cof_dir.createFile("config.toml", .{{}});
    defer config_file.close();

    const config_content = "[core]\n" ++ "block_size = 4096\n" ++ "hash_algorithm = \"blake3\"\n" ++ "cache_size_mb = 256\n" ++ "\n" ++ "[compression]\n" ++ "warm_threshold = 10\n" ++ "cold_threshold = 100\n" ++ "warm_level = 3\n" ++ "cold_level = 19\n" ++ "\n" ++ "[network]\n" ++ "protocol = \"udp\"\n" ++ "packet_size = 1400\n" ++ "timeout_ms = 5000\n" ++ "max_retries = 3\n" ++ "\n" ++ "[gc]\n" ++ "auto_gc = true\n" ++ "unreachable_days = 30\n";

    try config_file.writeAll(config_content);
    std.debug.print("Initialized empty cof repository in .cof/\n", .{{}});
}}

// Main entry point.
pub fn main() !void {{
    var gpa = std.heap.GeneralPurposeAllocator(.{{}}){{}};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {{
        std.debug.print("Usage: cof <command>\n", .{{}});
        return;
    }}

    const command = args[1];

    if (std.mem.eql(u8, command, "init")) {{
        try initRepository();
    }} else if (std.mem.eql(u8, command, "commit")) {{
        if (args.len < 4 or !std.mem.eql(u8, args[2], "-m")) {{
            std.debug.print("Usage: cof commit -m \"message\"\n", .{{}});
            return;
        }}
        const message = args[3];
        try commit(allocator, message);
    }} else {{
        std.debug.print("Unknown command: {s}\n", .{{command}});
    }}
}}
