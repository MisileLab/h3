const std = @import("std");
const rand = std.rand.DefaultPrng;
const stringl = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

pub fn main() !void {
    var args = std.process.args();
    _ = args.skip();
    var arg = args.next() orelse "bdsx";
    var rnd = rand.init(blk: {
        var seed: u64 = undefined;
        try std.os.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });

    var string: [100]u8 = undefined;
    for (&string) |*char| {
        char.* = stringl[rnd.random().intRangeLessThan(u8, 0, stringl.len-1)];
    }
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();
    try stdout.print("borg create ./backup::{s} ./{s}\n", .{string, arg});
    try bw.flush();
}