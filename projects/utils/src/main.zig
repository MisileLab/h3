const std = @import("std");
const rand = std.rand.DefaultPrng;
const io = std.io;
const stringl = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

// shortcut for comparing two string
pub fn c_s(a: [:0]const u8, b: [:0]const u8) bool {
    return std.mem.eql(u8, a, b);
}

pub fn main() !void {
    var args = std.process.args();
    _ = args.skip();
    const command = args.next() orelse "random";
    var bw = io.bufferedWriter(io.getStdOut().writer());
    const stdout = bw.writer();
    if (c_s(command, "random")) {
        var rnd = rand.init(blk: {
            var seed: u64 = undefined;
            try std.os.getrandom(std.mem.asBytes(&seed));
            break :blk seed;
        });

        var string: [100]u8 = undefined;
        for (&string) |*char| {
            char.* = stringl[rnd.random().intRangeLessThan(u8, 0, stringl.len - 1)];
        }
        try stdout.print("{s}\n", .{string});
    } else {
        try stdout.print("./utils random\n", .{});
    }
    try bw.flush();
}
