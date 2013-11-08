#!/usr/bin/perl

die "Usage: plotorder.pl <graphfile> <orderfile> <sizesfile>\n" unless ($#ARGV == 2);

$graphfile = shift(@ARGV);
$orderfile = shift(@ARGV);
$sizesfile = shift(@ARGV);


#=========================================================================
# Read the graph file
#=========================================================================
open(FPIN, "<$graphfile");
$_ = <FPIN>;
chomp($_);
($nvtxs, $nedges, $flags) = split(' ', $_);
$readvw = $flags&2;
$readew = $flags&1;

$nnz = 0;
$xadj[0] = 0;
for ($i=0; $i<$nvtxs; $i++) {
  $_ = <FPIN>;
  chomp($_);
  @fields = split(' ', $_);
  $vwgt[$i] = shift(@fields) if ($readvw);
  while (@fields) {
    $adjncy[$nnz] = shift(@fields)-1;
    $adjwgt[$nnz] = shift(@fields) if ($readew);
    $nnz++;
  }
  $xadj[$i+1] = $nnz;
}
close(FPIN);

