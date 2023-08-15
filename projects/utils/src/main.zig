const std = @import("std");
const rand = std.rand.DefaultPrng;
const io = std.io;
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
    var bw = io.bufferedWriter(io.getStdOut().writer());
    var br = io.bufferedReader(io.getStdIn().reader());
    const stdout = bw.writer();
    const stdin = br.reader();
    const palloc = std.heap.page_allocator;
    if (!(c_s(exe, "./utils") or c_s(exe, "utils"))) {
        const file = try std.fs.cwd().createFile("output.txt", .{ .read = true, .truncate = true });
        _ = try file.write(command);
        try bw.flush();
        var a = try stdin.readAllAlloc(palloc, 10);
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
