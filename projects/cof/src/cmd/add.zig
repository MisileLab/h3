const std = @import("std");
const core = @import("../core.zig");

pub fn run(allocator: std.mem.Allocator, paths: []const []const u8) !void {
    var index = try core.index.load(allocator);
    defer index.deinit();

    for (paths) |path| {
        std.debug.print("Adding: {s}\n", .{path});
        var hashes_managed = try core.objects.processPath(allocator, path);
        defer hashes_managed.deinit();

        // Convert to unmanaged for storage in the index
        const hashes_unmanaged = std.ArrayListUnmanaged([32]u8){
            .items = hashes_managed.items,
            .capacity = hashes_managed.capacity,
            .allocator = hashes_managed.allocator,
        };
        // Prevent deinit of the managed list from freeing the memory
        hashes_managed.items = &[_][32]u8{};

        var found = false;
        for (index.items) |*entry| {
            if (std.mem.eql(u8, entry.path, path)) {
                entry.*.deinit(allocator);
                entry.path = try allocator.dupe(u8, path);
                entry.hashes = hashes_unmanaged;
                found = true;
                break;
            }
        }

        if (!found) {
            try index.append(.{
                .path = try allocator.dupe(u8, path),
                .hashes = hashes_unmanaged,
            });
        }
    }

    try core.index.save(&index);
}