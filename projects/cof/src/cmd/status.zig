const std = @import("std");

pub fn run() !void {
    const cwd = std.fs.cwd();
    const stat = cwd.statFile(".cof") catch |err| {
        if (err == error.FileNotFound) {
            std.debug.print("Not a cof repository.\n", .{});
            return;
        }
        return err;
    };

    if (stat.kind != .directory) {
        std.debug.print("Not a cof repository.\n", .{});
        return;
    }

    std.debug.print("On branch main\n", .{});
    std.debug.print("nothing to commit, working tree clean\n", .{});
}

