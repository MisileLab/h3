const std = @import("std");

const IndexEntry = struct {
    path: []const u8,
    hashes: std.ArrayListUnmanaged([32]u8),

    pub fn deinit(self: *IndexEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        self.hashes.deinit(allocator);
    }
};

pub const Index = std.array_list.Managed(IndexEntry);

pub fn load(allocator: std.mem.Allocator) !Index {
    var index = Index.init(allocator);
    errdefer index.deinit();

    const cwd = std.fs.cwd();
    var index_dir = try cwd.openDir(".cof/index", .{});
    defer index_dir.close();

    var file = index_dir.openFile("staging", .{}) catch |err| {
        if (err == error.FileNotFound) return index;
        return err;
    };
    defer file.close();

    var reader_buffer: [1024]u8 = undefined;
    const file_reader = file.reader(&reader_buffer);
    const reader = file_reader.interface;

    while (true) {
        var path_len_buf: [2]u8 = undefined;
        reader.readNoEof(&path_len_buf) catch |err| {
            if (err == error.EndOfStream) break;
            return err;
        };
        const path_len = std.mem.readIntBig(u16, &path_len_buf, 0);

        const path = try allocator.alloc(u8, path_len);
        errdefer allocator.free(path);
        try reader.readNoEof(path);

        var num_blocks_buf: [4]u8 = undefined;
        try reader.readNoEof(&num_blocks_buf);
        const num_blocks = std.mem.readIntBig(u32, &num_blocks_buf, 0);

        var hashes = std.ArrayListUnmanaged([32]u8){};
        errdefer hashes.deinit(allocator);
        try hashes.ensureTotalCapacity(allocator, num_blocks);

        var i: u32 = 0;
        while (i < num_blocks) : (i += 1) {
            var hash: [32]u8 = undefined;
            try reader.readNoEof(&hash);
            try hashes.append(allocator, hash);
        }

        try index.append(allocator, .{ .path = path, .hashes = hashes });
    }

    return index;
}

pub fn save(index: *Index) !void {
    const cwd = std.fs.cwd();
    var index_dir = try cwd.openDir(".cof/index", .{});
    defer index_dir.close();

    var file = try index_dir.createFile("staging", .{});
    defer file.close();

    var writer_buffer: [1024]u8 = undefined;
    const file_writer = file.writer(&writer_buffer);
    const writer = file_writer.interface;

    for (index.items) |entry| {
        var path_len_buf: [2]u8 = undefined;
        std.mem.writeIntBig(u16, &path_len_buf, 0, @intCast(entry.path.len));
        try writer.writeAll(&path_len_buf);

        try writer.writeAll(entry.path);

        var num_blocks_buf: [4]u8 = undefined;
        std.mem.writeIntBig(u32, &num_blocks_buf, 0, @intCast(entry.hashes.items.len));
        try writer.writeAll(&num_blocks_buf);

        for (entry.hashes.items) |hash| {
            try writer.writeAll(&hash);
        }
    }

    try writer.flush();
}