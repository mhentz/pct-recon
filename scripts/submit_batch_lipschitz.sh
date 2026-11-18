#!/bin/bash -e

for phantom in {gammex,spiral,ctp528}; do
  for dataset in {ideal,realistic}; do
    for dose in {5,1,0.2}; do
        for n_projections in {720,45}; do

            pixel_w=$([ "$phantom" != "gammex" ] && echo 0.25 || echo 0.5)
            image_w=$([ "$phantom" != "gammex" ] && echo 880 || echo 700)

            ~/pct-recon/scripts/submit_lipschitz.sh \
                ${phantom}-${dataset} \
                $dose \
                $n_projections \
                $pixel_w \
                $image_w \
                25
        done
    done
  done
done
