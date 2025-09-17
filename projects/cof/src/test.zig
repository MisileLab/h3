const std = @import("std");

pub fn main() !void {
    // Write
    {
        var file = try std.fs.cwd().createFile("test.bin", .{});
        defer file.close();

        var writer_buffer: [1024]u8 = undefined;
        var file_writer = file.writer(&writer_buffer);
        const writer = &file_writer.interface;

        var int_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &int_buf, 12345, .big);
        try writer.*.writeAll(&int_buf);

        try writer.*.writeAll("hello");

        try writer.*.flush();
    }

    // Read
    {
        var file = try std.fs.cwd().openFile("test.bin", .{});
        defer file.close();

        var reader_buffer: [1024]u8 = undefined;
        var file_reader = file.reader(&reader_buffer);

        var int_buf: [4]u8 = undefined;
        try file_reader.readNoEof(&int_buf);
        const val = std.mem.readInt(u32, &int_buf, .big);
        std.debug.print("Read int: {d}\n", .{val});

        var str_buf: [5]u8 = undefined;
        try file_reader.readNoEof(&str_buf);
        std.debug.print("Read str: {s}\n", .{&str_buf});
    }
}
