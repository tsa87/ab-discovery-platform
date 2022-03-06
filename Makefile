# Install exact Python and CUDA versions
conda-update:
	conda env update --prune -f docs/environment.yaml
	echo "!!!RUN RIGHT NOW:\nconda activate ab-discovery"