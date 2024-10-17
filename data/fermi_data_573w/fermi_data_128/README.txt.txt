This directory contains Yitian's "Best Five" CZMS templates with an error in the pixel area adjustment.

Hey Ed, I noticed a mistake in the cartesian to healpix conversion I did before… the pixel area adjustment for exposure was written as cos(l) * cos(b) but it should just be cos(b). I’ll try to get you the correct bubbles templates and Return of the Templates templates ASAP. Sorry for the mistake!
6:43
This might affect the large l feature of the GP (that said model O is healpix and does not have this problem, so the mistake probably cant account for the GP’s behavior at large l