Matt and Prithvi currently have access. Feel free to fork the code to the internal repo under your org

See [README.md](http://README.md) for description of code, data, and schemas. Make sure to include @README.md into your Cursor chat whenever you’re asking general questions about the codebase (it’s written in a way to tell AI where in the repo to look for to answer your question)

# TODOs before production:

- in `.env`, change the `SIEVE_API_KEY` and pay for sievedata.com
- Make a data storage bucket for raw data (e.g. on cloudflare) and explain to crowdworkers how to upload the data there OR write the code to immediately start upload to the bucket e.g. when the user presses Ctrl+D in the terminal
- Change all the `PROD` variables to `True`
- Run `./build.sh` — that will build the app which you’ll then give to crowdworkers, it’ll appear in the `dist/` folder
    - Test that it actually works on windows+android. I only tested it on mac+android
- Make changes to the [Instructions for Data Collectors](https://www.notion.so/Instructions-for-Data-Collectors-217eefc8733380268679c3597593c3c9?pvs=21) page (marked with TODOs)

# Future improvement directions

- **postprocess** `action_analysis.json` and `frame_analysis.json` into a format that’s suitable for Roam’s DB or Alec’s tree generation model
    - probably just dump the entire json file as one prompt into gemini with 1M context and ask to summarize.
    - only do it once for every playthrough
- **make the game asset detection model better!**
    - segmentation model (SAM2 deployed [here](https://www.sievedata.com/functions/sieve/sam2/readme)) is near perfect, i.e. it’ll cut out objects perfectly
        
        ![CleanShot 2025-06-18 at 20.18.01@2x.png](attachment:171c8ec4-c0da-4645-9bbb-faf175c33b1b:CleanShot_2025-06-18_at_20.18.012x.png)
        
    - but Gemini which i use for finding objects often fucks up and i didn’t have time to make it much better
        - (reason: it was trained on real-world images, not cartoonish images - so when the game looks like a cartoon, it can’t come up with a good bounding box)
        - previously tried: yolo v8, florence-2 (both on Sieve). they’re even worse lol
        - using [this notebook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb) as basis for my implementation
    
    → TODO: postprocess all images in the `analysis/segmented_assets` folder, e.g. with another LLM call
    → OR ask the hired crowdworkers to delete incorrect images
    
    examples of what it thinks is a "Gold Coin":
    

![CleanShot 2025-06-18 at 20.21.06@2x.png](attachment:45340fa8-1325-483b-8011-f92aaa00b7e8:CleanShot_2025-06-18_at_20.21.062x.png)

![CleanShot 2025-06-18 at 20.21.31@2x.png](attachment:fcaf10f5-a6a6-4867-a6c6-aa5afcd2a851:CleanShot_2025-06-18_at_20.21.312x.png)

![CleanShot 2025-06-18 at 20.22.17@2x.png](attachment:02ae2355-2efd-4303-8291-2eddcdc22bce:CleanShot_2025-06-18_at_20.22.172x.png)

![CleanShot 2025-06-18 at 20.22.04@2x.png](attachment:b9956da6-24a3-4e1d-a390-cc155936dc44:CleanShot_2025-06-18_at_20.22.042x.png)

- **get useful data based on timestamps**
    - was outside of my 2-day scope but a lot of low hanging fruits
        - idea: use CV to see how fast the background or character moves based on user actions
        - idea: see by how much everything accelerates as the user progresses thru the game
        - idea: understand camera behavior, e.g. how much it tilts in response to user swipe
        - etc
    - use raw video data (`screen_recording.mp4`) and touch data (`touch_events.log`)
        - see `src.processing.visualize:parse_touch_log()` for log parsing
        - Caution: there are time mismatches between the android phone registering the touches and the video stream hitting the computer. VERY easy to fuck up! I killed like 4 hours reconciling these times. glhf
        - android docs: [1](https://source.android.com/docs/core/interaction/input/touch-devices#touchmajor-touchminor-toolmajor-toolminor-size-fields) [2](https://source.android.com/docs/core/interaction/input/getevent)
- **make analysis faster with more parallelism**
    - right now, you guys are a tier 1 org under Gemini
    - you’ll be allowed more requests per second as you progress thru tiers; adjust the concurrency accordingly (ask cursor it’ll understand)
    - or just cut frames from the video less frequently (see `src.processing.frame_cutter`)
- **if …_analysis.json files get too large**
    - summarize and shorten them with an LLM
    - or turn the json files into a RAG system
    
    → these files are just created with a prompt and a Pydantic BaseModel, it should be really easy to change if you want different data or want it to fit perfectly with your internal DB format