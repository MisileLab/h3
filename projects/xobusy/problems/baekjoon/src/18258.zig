const std = @import("std");
const mem = std.mem;

const Node = struct {
  value: u32,
  next_node: ?*Node
};

const Error = error{
  nullPointer
};

const Queue = struct {
  allocator: *const mem.Allocator,
  head: ?*Node,
  tail: ?*Node,
  size: u32,
  fn push(self: *Queue, value: u32) !void {
    self.size += 1;
    const node: *Node = try self.allocator.create(Node);
    node.* = Node{ .value = value, .next_node = null };
    if (self.tail != null) {
      const tail = self.tail orelse unreachable;
      tail.next_node = node;
    }
    self.tail = node;
    if (self.head == null) {
      self.head = node;
    }
  }
  fn pop(self: *Queue) !u32 {
    self.size -= 1;
    if (self.head == null) {
      return Error.nullPointer;
    }
    const head = self.head orelse unreachable;
    defer self.allocator.destroy(head);
    self.head = head.next_node;
    return head.value;
  }
};

fn cmp(v: []u8, v2: []const u8) bool {
  return std.mem.eql(u8, v, v2);
}

pub fn main() !void {
  const stdout = std.io.getStdOut().writer();
  const stdin = std.io.getStdIn().reader();
  const allocator = std.heap.page_allocator;

  var queue = Queue {
    .allocator = &allocator,
    .head = null,
    .tail = null,
    .size = 0,
  };

  const number = try std.fmt.parseInt(
    u32,
    try stdin.readUntilDelimiterAlloc(allocator, '\n', 7),
    10
  );
  var line: []u8 = undefined;

  for (0..number) |_| {
    line = try stdin.readUntilDelimiterAlloc(allocator, '\n', 11);
    if (cmp(line, "back")) {
      if (queue.tail == null) {
        try stdout.print("-1\n", .{});
      } else {
        const tail = queue.tail orelse unreachable;
        try stdout.print("{d}\n", .{tail.value});
      }
    } else if (cmp(line, "front")) {
      if (queue.head == null) {
        try stdout.print("-1\n", .{});
      } else {
        const head = queue.head orelse unreachable;
        try stdout.print("{d}\n", .{head.value});
      }
    } else if (cmp(line, "empty")) {
      try stdout.print("{d}\n", .{@intFromBool(queue.size == 0)});
    } else if (cmp(line, "size")) {
      try stdout.print("{d}\n", .{queue.size});
    } else if (cmp(line, "pop")) {
      if (queue.size == 0) {
        try stdout.print("-1\n", .{});
      } else {
        try stdout.print("{d}\n", .{try queue.pop()});
      }
    } else {
      const value = try std.fmt.parseInt(
        u32,
        std.mem.trimLeft(u8, line, "push "),
        10
      );
      try queue.push(value);
    }
  }
}

test "queue" {
  var allocator = std.heap.page_allocator;
  var queue = Queue{
    .allocator = &allocator,
    .head = null,
    .tail = null,
    .size = 0,
  };

  // Push elements into the queue
  try queue.push(10);
  try queue.push(20);
  try queue.push(30);

  // Verify size
  try std.testing.expect(queue.size == 3);

  // Pop elements and verify order
  try std.testing.expect(try queue.pop() == 10);
  try std.testing.expect(try queue.pop() == 20);
  try std.testing.expect(try queue.pop() == 30);

  // Verify size after pops
  try std.testing.expect(queue.size == 0);
}

