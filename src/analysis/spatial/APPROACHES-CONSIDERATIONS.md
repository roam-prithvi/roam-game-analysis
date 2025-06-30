## Approach

The current approach relies heavily on the video understanding baked into Gemini 2.5 Pro.

### Agent

Look at the [Spatial Reasoning](./SPATIAL_REASONING.md) doc to get more info but;
There's an agent that will look at each video chunk and then go in and edit the output JSON based on what it learnt at the moment.

## Scalable Improvements

I would say there's quite a lot of low hanging fruit that can be still extracted from this approach.

- Prompt optimizer step before the agent / LLM workflow even runs that builds out the prompt based on the game. This would be a one-two shot Gemini call that will load in the entire video as context and then based write a JSON like:

```json
{
  "camera_angle": {},
  "game_style": {},
  "assets": [],
  ...
  // more information like this that is essential to guide the 3D builder agent.
}
```

This then kicks off the agent.

- Have the agent look at the resultant Unity construction, look at the video again and then edit the 3D representation. This would be far better than any loop that just spits out the JSON. As can build out logic to view at the output, to iteratively improve each chunk representation.
  - We should ideally just output the final Unity representation and then have a tool to give the screenshot + angles of the scene to the agent.

## Alternative approach + ideas

- Single first pass to map out all the asset classes to give a shit about.
- Then pass through the chunks or the whole video again and again and edit each assets representation in 3D space.
- Some experimentations with SAM2 for each labelling different asset type might be interesting. If there are bounding boxes around the asset types, then that might help things a lot.
- What if for each asset type we're focusing on, we kinda paint it in a different color?
- The resultant JSON should not have colors, instead it should sorta just be same color, have another LLM call to paint things in. I believe this might boost accuracy because I think there's a intelligence budget that might be getting wasted in colors etc.

## Considerations

- Scalability across different types of games will not be easy from first impressions. Prompt tuning for each game will required atleast a little bit.
- From first impressions, general scalable spatial mapping seems to only be possible through large VLMs.
