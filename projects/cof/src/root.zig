const std = @import("std");

pub const StorageTier = enum {
    HOT,
    WARM,
    COLD,
};

pub const Block = struct {
    hash: [32]u8,
    data: []u8,
    tier: StorageTier,
    created_commit: u64,
    ref_count: u32,
};

pub const Commit = struct {
    id: [32]u8,
    parent: ?[32]u8,
    tree_root: [32]u8,
    timestamp: u64,
    author: []const u8,
    message: []const u8,
    sequence: u64,
};

pub const TreeEntry = struct {
    name: []const u8,
    mode: u32,
    hash: [32]u8,
    size: u64,
};

test "data structures alignment" {
    try std.testing.expect(@sizeOf(Block) > 32);
    try std.testing.expect(@sizeOf(Commit) > 32);
    try std.testing.expect(@sizeOf(TreeEntry) > 32);
}