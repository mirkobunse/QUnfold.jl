results/crab_protocols_nmd.tex: crab_protocols.jl
	julia --project=. $^ results/crab_protocols_Tp.tex $@
