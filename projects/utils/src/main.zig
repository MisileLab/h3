const std = @import("std");
const rand = std.rand.DefaultPrng;

pub fn main() !void {
    var rnd = rand.init(
        blk: {
            var seed: u64 = undefined;
            try std.os.getrandom(std.mem.asBytes(&seed));
            break :blk seed;
        }
    );
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();
    try stdout.print("borg create ./backup::{} ./bdsx\n", .{rnd.random().int(u128)});
    try bw.flush();
}
