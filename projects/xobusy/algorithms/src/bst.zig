const std = @import("std");
const mem = std.mem;

const Node = struct {
  value: usize,
  left: ?*Node,
  right: ?*Node,
  fn insert(self: *Node, allocator: *const mem.Allocator, element: usize) !void {
    if (element < self.value) {
      if (self.left == null) {
        const v = try allocator.create(Node);
        v.* = Node {
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
        const v = try allocator.create(Node);
        v.* = Node {
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
  fn search(self: *Node, allocator: *const mem.Allocator, element: usize) !bool {
    if (self.value == element) {
      return true;
    } else if (element < self.value) {
      if (self.left == null) {
        return false;
      } else {
        return try search(self.left orelse unreachable, allocator, element);
      }
    } else {
      if (self.right == null) {
        return false;
      } else {
        return try search(self.right orelse unreachable, allocator, element);
      }
    }
  }
};

fn cmp(v: []u8, v2: []const u8) bool {
  return std.mem.eql(u8, v, v2);
}

pub fn main() !void {
  const stdout = std.io.getStdOut().writer();
  const stdin = std.io.getStdIn().reader();
  const allocator = std.heap.page_allocator;
  const values = [_]usize{20, 40, 10, 25};

  const node = try allocator.create(Node);
  node.* = Node {
    .value = 30,
    .left = null,
    .right = null
  };

  for (values) |value| {try node.insert(&allocator, value);}

  while (true) {
    _ = try stdout.write("메뉴 선택\n1. 삽입 \n2. 검색\n");
    const resp = try stdin.readUntilDelimiterAlloc(allocator, '\n', 2);

    if (cmp(resp, "1")) {
      _ = try stdout.write("삽입할 값: ");
    } else if (cmp(resp, "2")) {
      _ = try stdout.write("검색할 값: ");
    }

    const value = try std.fmt.parseInt(
      usize, try stdin.readUntilDelimiterAlloc(allocator, '\n', 10), 10
    );

    if (cmp(resp, "1")) {
      try node.insert(&allocator, value);
//      _ = try stdout.print("{?}", .{node});
    } else if (cmp(resp, "2")) {
      if (try node.search(&allocator, value)) {
        _ = try stdout.write("값이 존재합니다.\n");
      } else {
        _ = try stdout.write("값이 존재하지 않습니다.\n");
      }
    }
  }
}

