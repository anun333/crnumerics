// philox4x32_10.v: Philox4x32-10 (Salmon, Moraes, Dror and Shaw, "Parallel random numbers: as easy as 1, 2, 3",
// SC 2011), the counter-based generator under crdist's samplers, one round per clock (2026-10-15).
//
// The block of four 32-bit words for a 128-bit counter {c3, c2, c1, c0} and a 64-bit key {k1, k0}: ten rounds,
// each multiplying c0 and c2 by the Philox constants and mixing in the key, the key bumped by the Weyl constants
// before every round but the first. It is the function crdist-sample.c's philox() computes, word for word, and
// crdist's stream (seed, substream) is this block at counter {substream >> 32, substream, j >> 32, j} under key
// {seed >> 32, seed} for words 4j to 4j + 3 (crdist.h, THE STREAM). hw/philox/test checks it against libcrdist.
//
// Interface: assert start for one clock with ctr and key; done rises for one clock ten clocks later, with the block
// on out ({w3, w2, w1, w0}, w0 in bits 31:0) until the next start. busy is high in between. Plain Verilog 2005.
// PHILOX_PLANTED_BUG (a define) changes one Weyl constant by one: the control the check must catch.
module philox4x32_10 (
  input  wire         clk,
  input  wire         rst,     // synchronous, active high
  input  wire         start,
  input  wire [127:0] ctr,     // {c3, c2, c1, c0}
  input  wire [63:0]  key,     // {k1, k0}
  output reg          busy,
  output reg          done,
  output reg  [127:0] out      // {w3, w2, w1, w0}
);
  localparam [31:0] M0 = 32'hD2511F53, M1 = 32'hCD9E8D57;
`ifdef PHILOX_PLANTED_BUG
  localparam [31:0] W0 = 32'h9E3779B8, W1 = 32'hBB67AE85;
`else
  localparam [31:0] W0 = 32'h9E3779B9, W1 = 32'hBB67AE85;
`endif

  reg [31:0] c0, c1, c2, c3, k0, k1;
  reg [3:0]  r;

  // one round of the current state
  wire [63:0] p0 = M0 * c0;
  wire [63:0] p1 = M1 * c2;
  wire [31:0] n0 = p1[63:32] ^ c1 ^ k0;
  wire [31:0] n1 = p1[31:0];
  wire [31:0] n2 = p0[63:32] ^ c3 ^ k1;
  wire [31:0] n3 = p0[31:0];

  always @(posedge clk) begin
    done <= 1'b0;
    if (rst) begin
      busy <= 1'b0;
      r <= 4'd0;
    end else if (start) begin
      {c3, c2, c1, c0} <= ctr;
      {k1, k0} <= key;
      r <= 4'd0;
      busy <= 1'b1;
    end else if (busy) begin
      {c3, c2, c1, c0} <= {n3, n2, n1, n0};
      if (r == 4'd9) begin
        out <= {n3, n2, n1, n0};
        busy <= 1'b0;
        done <= 1'b1;
      end else begin
        k0 <= k0 + W0;
        k1 <= k1 + W1;
        r <= r + 4'd1;
      end
    end
  end
endmodule
