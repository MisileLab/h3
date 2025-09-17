const std = @import("std");
const core = @import("../core.zig");

fn getFileList(allocator: std.mem.Allocator, file_buffer: *[1024][]const u8, count: *usize) !void {
    // 1. Load ignore patterns
    var ignore_patterns: [128][]const u8 = undefined;
    var ignore_count: usize = 0;
    if (std.fs.cwd().openFile(".cofignore", .{})) |file| {
        defer file.close();
        const stat = try file.stat();
        const content = try allocator.alloc(u8, stat.size);
        defer allocator.free(content);
        _ = try file.readAll(content);

        var it = std.mem.splitSequence(u8, content, "\n");
        while (it.next()) |line| {
            if (ignore_count >= 128) break;
            const trimmed = std.mem.trim(u8, line, " \r");
            if (trimmed.len == 0) continue;
            ignore_patterns[ignore_count] = try allocator.dupe(u8, trimmed);
            ignore_count += 1;
        }
    } else |err| {
        if (err != error.FileNotFound) {
            return err;
        }
    }
    defer {
        for (ignore_patterns[0..ignore_count]) |p| allocator.free(p);
    }

    // 2. Walk directory
    var dir = try std.fs.cwd().openDir(".", .{ .iterate = true });
    defer dir.close();
    var walker = try dir.walk(allocator);
    defer walker.deinit();

    outer: while (try walker.next()) |entry| {
        if (std.mem.startsWith(u8, entry.path, ".cof")) {
            continue :outer;
        }

        for (ignore_patterns[0..ignore_count]) |pattern| {
            if (std.mem.endsWith(u8, pattern, "/")) {
                const dir_name = pattern[0 .. pattern.len - 1];
                if (std.mem.startsWith(u8, entry.path, dir_name)) {
                    continue :outer;
                }
            } else if (std.mem.startsWith(u8, pattern, "*.")) {
                if (entry.kind == .file and std.mem.endsWith(u8, entry.path, pattern[1..])) {
                    continue :outer;
                }
            }
        }

        if (entry.kind == .file) {
            if (count.* >= 1024) break;
            if (std.mem.eql(u8, entry.path, ".cofignore")) continue;
            if (std.mem.eql(u8, entry.path, ".gitignore")) continue;

            file_buffer[count.*] = try allocator.dupe(u8, entry.path);
            count.* += 1;
        }
    }
}

// Executes the commit process.
pub fn run(allocator: std.mem.Allocator, message: []const u8) !void {
    _ = message;

    var file_buffer: [1024][]const u8 = undefined;
    var file_count: usize = 0;
    try getFileList(allocator, &file_buffer, &file_count);

    const file_list = file_buffer[0..file_count];

    defer {
        for (file_list) |item| {
            allocator.free(item);
        }
    }

    for (file_list) |file_path| {
        std.debug.print("Processing: {s}\n", .{
            file_path
        });
        var hashes = try core.objects.processPath(allocator, file_path);
        defer hashes.deinit();
        std.debug.print("  Blocks: {d}\n", .{
            hashes.items.len
        });
    }

    std.debug.print("Commit successful (object generation only).\n", .{} );
}
