// tb_kat.v: Random123's three known answers through philox4x32_10.v in a second simulator (Icarus Verilog), so the
// result does not rest on Verilator alone (2026-10-15). Prints PASS or FAIL per answer and the count.
`timescale 1ns/1ps
module tb_kat;
  reg clk = 0, rst = 1, start = 0;
  reg [127:0] ctr;
  reg [63:0] key;
  wire busy, done;
  wire [127:0] out;
  integer pass = 0, t, lat;
  reg [127:0] want;

  philox4x32_10 dut (.clk(clk), .rst(rst), .start(start), .ctr(ctr), .key(key), .busy(busy), .done(done), .out(out));
  always #5 clk = ~clk;

  task run(input [127:0] c, input [63:0] k, input [127:0] w);
    begin
      ctr = c; key = k; want = w;
      @(negedge clk) start = 1;
      @(negedge clk) start = 0;
      lat = 0;   // clocks counted from the first round, as tb.cpp counts them
      while (!done && lat < 40) begin @(negedge clk); lat = lat + 1; end
      if (done && out == want && lat == 10) begin pass = pass + 1; $display("PASS %h", out); end
      else $display("FAIL %h want %h after %0d clocks", out, want, lat);
    end
  endtask

  initial begin
    @(negedge clk); @(negedge clk) rst = 0;
    // {c3, c2, c1, c0}, {k1, k0}, {w3, w2, w1, w0}
    run(128'h0, 64'h0, {32'h9b00dbd8, 32'hbc57ac4c, 32'he169c58d, 32'h6627e8d5});
    run({4{32'hffffffff}}, {2{32'hffffffff}}, {32'h6d5451fd, 32'ha20bc7c6, 32'h41c83b0e, 32'h408f276d});
    run({32'h03707344, 32'h13198a2e, 32'h85a308d3, 32'h243f6a88}, {32'h299f31d0, 32'ha4093822},
        {32'h24126ea1, 32'h5001e420, 32'h94fdcceb, 32'hd16cfe09});
    $display("known answers %0d of 3", pass);
    $finish;
  end
endmodule
