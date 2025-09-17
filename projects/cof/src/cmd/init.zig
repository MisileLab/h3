const std = @import("std");

pub fn run() !void {
    const fs = std.fs;
    const cwd = fs.cwd();

    try cwd.makeDir(".cof");
    var cof_dir = try cwd.openDir(".cof", .{
    });
    defer cof_dir.close();

    try cof_dir.makeDir("objects");
    var objects_dir = try cof_dir.openDir("objects", .{
    });
    defer objects_dir.close();
    try objects_dir.makeDir("hot");
    try objects_dir.makeDir("warm");
    try objects_dir.makeDir("cold");

    try cof_dir.makeDir("index");
    try cof_dir.makeDir("refs");
    var refs_dir = try cof_dir.openDir("refs", .{
    });
    defer refs_dir.close();
    try refs_dir.makeDir("heads");

    try cof_dir.makeDir("locks");

    var config_file = try cof_dir.createFile("config.toml", .{
    });
    defer config_file.close();

    const config_content = "[core]\n" ++ "block_size = 4096\n" ++ "hash_algorithm = \"blake3\"\n" ++ "cache_size_mb = 256\n" ++ "\n" ++ "[compression]\n" ++ "warm_threshold = 10\n" ++ "cold_threshold = 100\n" ++ "warm_level = 3\n" ++ "cold_level = 19\n" ++ "\n" ++ "[network]\n" ++ "protocol = \"udp\"\n" ++ "packet_size = 1400\n" ++ "timeout_ms = 5000\n" ++ "max_retries = 3\n" ++ "\n" ++ "[gc]\n" ++ "auto_gc = true\n" ++ "unreachable_days = 30\n";

    try config_file.writeAll(config_content);
    std.debug.print("Initialized empty cof repository in .cof/\n", .{
    });
}
