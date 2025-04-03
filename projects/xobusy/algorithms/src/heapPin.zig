const std = @import("std");

pub fn buildHeap(arr: []usize, n: usize) !void {
  var v = n/2-1;
  std.debug.print("{}", .{v});
  while (v >= 0) {
    try heapify(arr, v, n);
    if (v == 0) {break;}
    v -= 1;
  }
  for (v..0) |i| {
    try heapify(arr, i, n);
  }
}

pub fn swap(arr: []usize, i: usize, j: usize) !void {
  const temp = arr[i];
  arr[i] = arr[j];
  arr[j] = temp;
}

pub fn heapify(arr: []usize, k: usize, n: usize) !void {
  var largest = k;
  const left = 2*k+1;
  const right = 2*k+2;
  if (left < n and arr[left] > arr[largest]) {
    largest = left;
  }
  if (right < n and arr[right] > arr[largest]) {
    largest = right;
  }
  std.debug.print("heapify: {d} {d} {d}\n", .{k, largest, n});
  if (largest == k) {return;}
  try swap(arr, k, largest);
  try heapify(arr, largest, n);
}

pub fn main() !void {
  const stdout = std.io.getStdOut().writer();
  var data = [_]usize{10, 20, 14, 23, 11, 50, 30, 34, 9};
  
  try buildHeap(&data, data.len);

  try stdout.print("{any}", .{data});
}

