# Examples

Real RD Flow 64 outputs at 64 x 64, made from short requests that Retro Diffusion's caption writer expanded into full
captions. `examples.json` has each request and its caption, so you can run them yourself:

```bash
python generate.py "$(python -c "import json; print(json.load(open('examples/examples.json'))[0]['caption'])")"
```

- `dragon.png`: a small red dragon sitting (transparent)
- `owl.png`: a round brown owl on a twig (transparent)
- `astronaut.png`: an astronaut waving (transparent)
- `pirate-ship.png`: a pirate ship (transparent)
- `robot.png`: a friendly little robot (transparent)
- `flaming-sword.png`: a sword with a flaming blade (transparent)
- `crystals.png`: a cluster of purple crystals (transparent)
- `chef-cat.png`: an orange cat dressed as a chef (transparent)
- `mushroom-frog.png`: a frog sitting on a mushroom (transparent)
- `ramen.png`: a bowl of ramen (transparent)
- `lighthouse.png`: a red and white lighthouse (transparent)
- `hot-air-balloon.png`: a hot air balloon (transparent)
- `wizard-cat.png`: a black cat wearing a wizard hat (transparent)
- `cactus.png`: a potted cactus with a pink flower (transparent)
- `desert-oasis.png`: a desert oasis with palm trees (opaque, add --opaque)
- `coral-reef.png`: an underwater coral reef with fish (opaque, add --opaque)
- `night-city.png`: a city skyline at night (opaque, add --opaque)
- `mountain-lake.png`: a mountain lake at sunrise (opaque, add --opaque)
- `cozy-bedroom.png`: a cozy bedroom at night (opaque, add --opaque)
- `space-station.png`: a space station orbiting a planet (opaque, add --opaque)
- `volcano.png`: an erupting volcano at dusk (opaque, add --opaque)

`red-potion.png` and `autumn-forest.png` are the two worked examples in [PROMPTING.md](../PROMPTING.md).
