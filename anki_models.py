import genanki
import settings

# Unified Image Occlusion Enhanced+ model
IOE_MODEL = genanki.Model(
    1876543210,
    settings.IOE_MODEL_NAME,
    fields=[
        {'name': settings.IOE_FIELDS['qmask']},
        {'name': settings.IOE_FIELDS['omask']},
    ],
    templates=[{
        'name': 'IO Card',
        'qfmt': f"{{{{{settings.IOE_FIELDS['qmask']}}}}}",
        'afmt': f"{{{{{settings.IOE_FIELDS['omask']}}}}}",
    }],
    css="""
#io-original { display: block; margin: 0 auto; }
"""
) 