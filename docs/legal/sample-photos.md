# Sample photograph credits

These three photographs are real, unedited source photographs used as bundled application demonstrations. They are individually covered by the [Pexels License](https://www.pexels.com/license/), identified as `LicenseRef-Pexels` in the provenance manifest. They are **not** relicensed under the code's AGPL or described as public-domain/CC0 assets.

| Bundled image | Photographer and original page |
| --- | --- |
| `utility_rear_view_01.jpg` | [Jan Zakelj — utility worker from behind](https://www.pexels.com/photo/back-view-of-a-person-wearing-white-hard-hat-and-reflectorize-jacket-13182107/) |
| `waterfront_rear_view_01.jpg` | [Nazmul Haque — waterfront crew](https://www.pexels.com/photo/construction-workers-at-waterfront-site-35861510/) |
| `piling_rear_view_01.jpg` | [MO ZHOU — piling works](https://www.pexels.com/photo/man-in-uniform-working-at-construction-site-4311990/) |

Pexels permits website/app use. Its separate terms restrict stock-photo/wallpaper redistribution, selling unchanged copies, implied endorsement, trademark use and offensive portrayal of identifiable people. Follow the linked license when reusing the images; they are not a stock-photo collection offered for redistribution.

The people and organizations pictured do not endorse this project. Demonstration boundaries are invented configuration examples, not claims about the actual site's rules or the workers' conduct. Model observations can be wrong; these photos are not an accuracy benchmark or training-label set.

Each exact source file was reviewed on 2026-09-12: all visible workers face away from the camera, with no visible recognizable face. `examples/demo_scenes.yaml` pins each image hash. Only the matching, explicitly selected sample can use the “Keep rear-view sample clear” option. Uploaded images, videos and cameras still use their own privacy controls. The privacy reason and actual redaction state are saved in the run manifest.

Source URLs, hashes and original license references are recorded in [provenance.yaml](provenance.yaml). The blank patch on the utility worker's helmet was present in the photographer-provided original; it was not added or removed here. Screenshot compositions include AGPL application UI and the separately licensed photographs.

## Hong Kong construction gallery

Two additional photographs are credited to [Catgirlmutant](https://unsplash.com/@catgirlmutant) and retain the [Unsplash License](https://unsplash.com/license), recorded as `LicenseRef-Unsplash`. They are application examples, not a competing image library, and are not relicensed AGPL.

- `hk_site_access_01.jpg`: [Hong Kong building site](https://unsplash.com/photos/a-man-wearing-a-hard-hat-BRQQtnsG2do).
- `hk_tsuen_wan_scaffolding_01.jpg`: [Tsuen Wan, Hong Kong](https://unsplash.com/photos/a-man-on-a-scaffold-working-on-a-building-PW-jyG50hpc).

These locations come from the original photographer pages. The README gallery uses publicly licensed photographs with redaction explicitly disabled; that does not claim that no faces are visible. The normal upload/live defaults and the exact-hash rear-view sample exception are unchanged. The original city for the Pexels piling photo has not been verified; Chinese site signage alone is not treated as geolocation evidence.

`tools/render_readme_gallery.py` runs the actual bundled detector, copies its annotations, and records source/model/output hashes. It does not draw invented detections or retouch the scene.
