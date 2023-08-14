const std = @import("std");
const rand = std.rand.DefaultPrng;
const stringl = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

// shortcut for comparing two string
pub fn c_s(a: [:0]const u8, b: [:0] const u8) bool {
    return std.mem.eql(u8, a, b);
}

pub fn main() !void {
    var args = std.process.args();
    var exe = args.next() orelse "utils";
    var command = args.next() orelse "backup";
    var arg = args.next() orelse "bdsx";
    const stdout_file = std.io.getStdOut().writer();
    const stdin_file = std.io.getStdIn().reader();
    var bw = std.io.bufferedWriter(stdout_file);
    var br = std.io.bufferedReader(stdin_file);
    const stdout = bw.writer();
    const stdin = br.reader();
    if (!(c_s(exe, "./utils") or c_s(exe, "utils"))) {
        try stdout.print("{s}", .{command});
        try bw.flush();
        var a = try stdin.readAllAlloc(std.heap.page_allocator, 10);
        _ = try stdin.read(a);
    }
    else if (c_s(command, "backup")) {
        var rnd = rand.init(blk: {
            var seed: u64 = undefined;
            try std.os.getrandom(std.mem.asBytes(&seed));
            break :blk seed;
        });

        var string: [100]u8 = undefined;
        for (&string) |*char| {
            char.* = stringl[rnd.random().intRangeLessThan(u8, 0, stringl.len - 1)];
        }
        try stdout.print("borg create --progress ./backup::{s} ./{s}\n", .{ string, arg });
        try bw.flush();
    } else {
        try stdout.print("./utils backup\n", .{});
        try bw.flush();
    }
}
