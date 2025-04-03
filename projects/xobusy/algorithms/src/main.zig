const std = @import("std");

pub fn main() !void {
  const stdout_file = std.io.getStdOut().writer();
  var bw = std.io.bufferedWriter(stdout_file);
  const stdout = bw.writer();

  try stdout.print("hi", .{});

  try bw.flush(); // Don't forget to flush!
}

