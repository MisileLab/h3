const std = @import("std");

pub fn bytesToHex(bytes: []const u8, out: []u8) void {
    const hex_chars = "0123456789abcdef";
    for (bytes, 0..) |b, i| {
        out[i * 2] = hex_chars[b >> 4];
        out[i * 2 + 1] = hex_chars[b & 0x0F];
    }
}

pub fn processPath(allocator: std.mem.Allocator, path: []const u8) !std.array_list.Managed([32]u8) {
    var block_hashes = std.array_list.Managed([32]u8).init(allocator);
    errdefer block_hashes.deinit();

    const block_size = 4096;
    var buffer: [block_size]u8 = undefined;
    var hex_buffer: [64]u8 = undefined;

    const cwd = std.fs.cwd();
    var objects_dir = try cwd.openDir(".cof/objects/hot", .{});
    defer objects_dir.close();

    var file = try cwd.openFile(path, .{});
    defer file.close();

    while (true) {
        const bytes_read = try file.read(&buffer);
        if (bytes_read == 0) break;
        const chunk = buffer[0..bytes_read];

        var hasher = std.crypto.hash.Blake3.init(.{});
        hasher.update(chunk);
        var hash: [32]u8 = undefined;
        hasher.final(&hash);

        try block_hashes.append(hash);

        bytesToHex(&hash, &hex_buffer);
        const hash_hex = hex_buffer[0..];

        _ = objects_dir.statFile(hash_hex) catch |err| {
            if (err == error.FileNotFound) {
                var block_file = try objects_dir.createFile(hash_hex, .{});
                defer block_file.close();
                try block_file.writeAll(chunk);
            } else {
                return err;
            }
        };
    }
    return block_hashes;
}
