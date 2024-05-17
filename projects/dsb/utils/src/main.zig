const std = @import("std");
const rand = std.crypto.random;
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
