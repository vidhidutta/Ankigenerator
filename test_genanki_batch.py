import genanki
from PIL import Image, ImageDraw

# 1. Prepare a minimal IOE note model
IOE_MODEL = genanki.Model(
  1607392319, 'Test IOE Model',
  fields=[{'name': 'Question'}, {'name': 'Answer'}],
  templates=[{
    'name': 'Card 1',
    'qfmt': '{{Question}}',
    'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
  }]
)

# 2. Create dummy images and notes
deck = genanki.Deck(2059400110, 'Batch Test Deck')
for i in range(100):
    # tiny 50×50 red box on clear bg
    img = Image.new("RGBA", (50, 50), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10,10,40,40], fill=(255,0,0,128), outline=(255,0,0,255), width=2)
    q_path = f"qmask_test_{i}.png"
    o_path = f"omask_test_{i}.png"
    img.save(q_path)
    img.save(o_path)
    # Note with two fields pointing to those images
    note = genanki.Note(
      model=IOE_MODEL,
      fields=[f"<img src='{q_path}'>", f"<img src='{o_path}'>"]
    )
    deck.add_note(note)

# 3. Export to .apkg
package = genanki.Package(deck, media_files=[f"qmask_test_{i}.png" for i in range(100)] +
                                      [f"omask_test_{i}.png" for i in range(100)])
package.write_to_file('batch_test.apkg')
print("Exported batch_test.apkg")
