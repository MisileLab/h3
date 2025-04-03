const std = @import("std");
const mem = std.mem;

const Element = struct {
  value: usize,
  left: ?*Element,
  right: ?*Element,
  fn insert(self: *Element, allocator: *const mem.Allocator, element: usize) !void {
    if (element < self.value) {
      if (self.left == null) {
        const v = try allocator.create(Element);
        v.* = Element {
          .value = element,
          .left = null,
          .right = null
        };
        self.left = v;
      } else {
        try insert(self.left orelse unreachable, allocator, element);
      }
    } else {
      if (self.right == null) {
        const v = try allocator.create(Element);
        v.* = Element {
          .value = element,
          .left = null,
          .right = null
        };
        self.right = v;
      } else {
        try insert(self.right orelse unreachable, allocator, element);
      }
    }
  }
};

pub fn main() !void {
  const data = [_]usize{10, 20, 14, 23, 11, 50, 30, 34, 9};
  const stdout = std.io.getStdOut().writer();
  const allocator = std.heap.page_allocator;

  const heap = try allocator.create(Element);
  heap.* = Element {
    .value = data[0],
    .left = null,
    .right = null
  };

  for (data) |value| {
    try heap.insert(&allocator, value);
  }

  try stdout.print("{?}", .{heap});
}

