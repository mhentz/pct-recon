#!/bin/bash

# ----------------------------------------------------------------------------
#  TV
# ----------------------------------------------------------------------------

# rerun alpha 5e-2 reconstructions
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh ctp528-ideal 1 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh ctp528-ideal 5 45 0.25 200 zero

# ~/pct-recon/scripts/submit_multiple_fista_tv.sh ctp528-realistic 5 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh ctp528-realistic 5 45 0.25 200 zero

# ~/pct-recon/scripts/submit_multiple_fista_tv.sh gammex-realistic 5 720 0.5 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh gammex-realistic 5 45 0.5 200 zero

# ~/pct-recon/scripts/submit_multiple_fista_tv.sh spiral-ideal 5 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh spiral-ideal 5 45 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh spiral-ideal 5 15 0.25 200 zero

# ~/pct-recon/scripts/submit_multiple_fista_tv.sh spiral-realistic 5 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh spiral-realistic 5 45 0.25 200 zero


# run spiral and ctp528 realistic TV reconstructions at 1 mGy
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh spiral-realistic 1 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh spiral-realistic 1 45 0.25 200 zero

# ~/pct-recon/scripts/submit_multiple_fista_tv.sh gammex-realistic 1 720 0.5 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh gammex-realistic 1 45 0.5 200 zero

# ~/pct-recon/scripts/submit_multiple_fista_tv.sh ctp528-realistic 1 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh ctp528-realistic 1 45 0.25 200 zero



# run realistic TV reconstructions at 0.2 mGy
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh spiral-ideal 0.2 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh spiral-ideal 0.2 45 0.25 200 zero

# ~/pct-recon/scripts/submit_multiple_fista_tv.sh spiral-realistic 0.2 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh spiral-realistic 0.2 45 0.25 200 zero

# ~/pct-recon/scripts/submit_multiple_fista_tv.sh gammex-realistic 0.2 720 0.5 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh gammex-realistic 0.2 45 0.5 200 zero

# ~/pct-recon/scripts/submit_multiple_fista_tv.sh ctp528-realistic 0.2 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_tv.sh ctp528-realistic 0.2 45 0.25 200 zero

# ----------------------------------------------------------------------------
#  dTV
# ----------------------------------------------------------------------------
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh ctp528-ideal 5 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh ctp528-ideal 5 45 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh ctp528-ideal 1 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh ctp528-ideal 1 45 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh ctp528-ideal 0.2 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh ctp528-ideal 0.2 45 0.25 200 zero

# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh gammex-realistic 5 720 0.5 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh gammex-realistic 5 45 0.5 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh spiral-realistic 5 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh spiral-realistic 5 45 0.25 200 zero
~/pct-recon/scripts/submit_multiple_fista_dtv.sh ctp528-realistic 5 720 0.25 200 zero
~/pct-recon/scripts/submit_multiple_fista_dtv.sh ctp528-realistic 5 45 0.25 200 zero


# ~/pct-recon/scripts/submit_fista.sh gammex-realistic 5 45 0.5 dTV 1.58e-3 1e-2 200 zero
# ~/pct-recon/scripts/submit_fista.sh gammex-realistic 5 45 0.5 dTV 3.54e-3 1e-1 200 zero
# ~/pct-recon/scripts/submit_fista.sh gammex-realistic 5 45 0.5 dTV 3.54e-3 2e-1 200 zero

# ~/pct-recon/scripts/submit_fista.sh gammex-realistic 5 720 0.5 dTV 1.58e-3 1e-1 200 zero
# ~/pct-recon/scripts/submit_fista.sh gammex-realistic 5 720 0.5 dTV 1.58e-3 2e-1 200 zero


# ~/pct-recon/scripts/submit_fista.sh spiral-realistic 5 45 0.25 dTV 1.58e-3 2e-1 200 zero
# ~/pct-recon/scripts/submit_fista.sh spiral-realistic 5 45 0.25 dTV 1e-3 1e-1 200 zero
# ~/pct-recon/scripts/submit_fista.sh spiral-realistic 5 45 0.25 dTV 1e-3 1e-2 200 zero
# ~/pct-recon/scripts/submit_fista.sh spiral-realistic 5 45 0.25 dTV 2.5e-3 2e-1 200 zero
# ~/pct-recon/scripts/submit_fista.sh spiral-realistic 5 45 0.25 dTV 3.54e-3 1e-1 200 zero

# ~/pct-recon/scripts/submit_fista.sh spiral-realistic 5 720 0.25 dTV 3.54e-3 2e-1 200 zero
# ~/pct-recon/scripts/submit_fista.sh spiral-realistic 5 720 0.25 dTV 1.58e-3 1e-2 200 zero


# ~/pct-recon/scripts/submit_fista.sh ctp528-realistic 5 45 0.25 dTV 1.58e-3 1e-2 200 zero
# ~/pct-recon/scripts/submit_fista.sh ctp528-realistic 5 45 0.25 dTV 1e-3 2e-1 200 zero
# ~/pct-recon/scripts/submit_fista.sh ctp528-realistic 5 45 0.25 dTV 1e-3 2e-1 200 zero
# ~/pct-recon/scripts/submit_fista.sh ctp528-realistic 5 45 0.25 dTV 2.5e-3 1e-1 200 zero
# ~/pct-recon/scripts/submit_fista.sh ctp528-realistic 5 45 0.25 dTV 2.5e-3 2e-1 200 zero
# ~/pct-recon/scripts/submit_fista.sh ctp528-realistic 5 45 0.25 dTV 3.54e-3 2e-1 200 zero

# ~/pct-recon/scripts/submit_fista.sh ctp528-realistic 5 720 0.25 dTV 1.58e-3 1e-1 200 zero
# ~/pct-recon/scripts/submit_fista.sh ctp528-realistic 5 720 0.25 dTV 1.58e-3 2e-1 200 zero
# ~/pct-recon/scripts/submit_fista.sh ctp528-realistic 5 720 0.25 dTV 1e-3 1e-1 200 zero
# ~/pct-recon/scripts/submit_fista.sh ctp528-realistic 5 720 0.25 dTV 3.54e-3 2e-1 200 zero


# ~/pct-recon/scripts/submit_fista.sh ctp528-realistic 0.2 45 0.25 dTV 1.58e-3 1e-1 200 zero
# ~/pct-recon/scripts/submit_fista.sh spiral-realistic 0.2 720 0.5 dTV 3.54e-3 1e-1 200 zero



# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh gammex-realistic 1 720 0.5 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh gammex-realistic 1 45 0.5 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh spiral-realistic 1 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh spiral-realistic 1 45 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh ctp528-realistic 1 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh ctp528-realistic 1 45 0.25 200 zero

# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh gammex-realistic 0.2 720 0.5 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh gammex-realistic 0.2 45 0.5 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh spiral-realistic 0.2 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh spiral-realistic 0.2 45 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh ctp528-realistic 0.2 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh ctp528-realistic 0.2 45 0.25 200 zero

# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh gammex-ideal 5 720 0.5 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh gammex-ideal 5 45 0.5 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh gammex-ideal 1 720 0.5 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh gammex-ideal 1 45 0.5 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh gammex-ideal 0.2 720 0.5 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh gammex-ideal 0.2 45 0.5 200 zero

# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh spiral-ideal 5 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh spiral-ideal 5 45 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh spiral-ideal 1 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh spiral-ideal 1 45 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh spiral-ideal 0.2 720 0.25 200 zero
# ~/pct-recon/scripts/submit_multiple_fista_dtv.sh spiral-ideal 0.2 45 0.25 200 zero
