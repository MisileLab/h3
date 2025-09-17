const std = @import("std");

const init = @import("cmd/init.zig");
const commit = @import("cmd/commit.zig");
const status = @import("cmd/status.zig");
const add = @import("cmd/add.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: cof <command>\n", .{});
        return;
    }

    const command = args[1];

    if (std.mem.eql(u8, command, "init")) {
        try init.run();
    } else if (std.mem.eql(u8, command, "commit")) {
        if (args.len < 4 or !std.mem.eql(u8, args[2], "-m")) {
            std.debug.print("Usage: cof commit -m \"message\"\n", .{});
            return;
        }
        const message = args[3];
        try commit.run(allocator, message);
    } else if (std.mem.eql(u8, command, "status")) {
        try status.run();
    } else if (std.mem.eql(u8, command, "add")) {
        if (args.len < 3) {
            std.debug.print("Usage: cof add <file>...\n", .{});
            return;
        }
        try add.run(allocator, args[2..]);
    } else {
        std.debug.print("Unknown command: {s}\n", .{command});
    }
}
