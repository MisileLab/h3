const std = @import("std");
const rand = std.crypto.random;
const io = std.io;
const process = std.process;
const stringl = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

// shortcut for comparing two string
pub fn c_s(a: [:0]const u8, b: [:0]const u8) bool {
    return std.mem.eql(u8, a, b);
}

pub fn main() !void {
    var alloc = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = alloc.deinit();
    const allocator = alloc.allocator();

    const args = try process.argsAlloc(allocator);
    defer process.argsFree(allocator, args);
    const command = if (args.len == 1) "random" else args[1];

    var bw = io.bufferedWriter(io.getStdOut().writer());
    const stdout = bw.writer();
    if (c_s(command, "random")) {
        var string: [100]u8 = undefined;
        for (&string) |*char| {
            char.* = stringl[rand.uintLessThan(u8, stringl.len - 1)];
        }
        try stdout.print("{s}\n", .{string});
    } else {
        try stdout.print("./utils random\n", .{});
    }
    try bw.flush();
}
