# Cursor Verification Prompt

## Context
You are reviewing an Anki flashcard generator application that had several critical issues. The user reported seeing errors in a screenshot including "expected string or bytes-like object, got 'dict'", "No valid flashcards", and "15 invalid flashcards skipped". 

## Issues to Verify

### 1. Template Formatting Error
**Issue**: `PROMPT_TEMPLATE.format()` was receiving a dictionary for `{flashcard_type}` but expected a string.

**Check these files**:
- `flashcard_generator.py`: Look for `get_flashcard_type_string()` function
- Verify all `PROMPT_TEMPLATE.format()` calls use `FLASHCARD_TYPE_STRING` instead of `stringify_dict(FLASHCARD_TYPE)`
- Check both semantic processing and basic processing paths

**Expected fixes**:
- Function `get_flashcard_type_string()` should convert `FLASHCARD_TYPE` dictionary to string like "Level 1 and Level 2"
- All template formatting should use `FLASHCARD_TYPE_STRING` variable
- No more `stringify_dict(FLASHCARD_TYPE)` in template formatting

### 2. Basic Processing Fallback
**Issue**: `generate_multimodal_flashcards_http()` was creating empty flashcards instead of calling the API.

**Check these files**:
- `flashcard_generator.py`: Look for `generate_multimodal_flashcards_http()` function

**Expected fixes**:
- Function should actually call OpenAI API with proper requests
- Should include proper error handling and timeouts
- Should process images correctly with file existence checks
- Should parse AI responses and create real Flashcard objects
- Should include rate limiting (`time.sleep(0.1)`)

### 3. Audio File Handling
**Issue**: Audio file paths might be directories instead of files.

**Check these files**:
- `gradio_interface.py`: Look for `find_audio_file()` function
- `gradio_interface_fixed.py`: Look for `find_audio_file()` function

**Expected fixes**:
- Function should check if path is a file (return as-is)
- Function should check if path is a directory (search for audio files inside)
- Should support multiple audio formats (.mp3, .wav, .m4a, .flac, .aac, .ogg)
- Should return first audio file found or None
- Should be integrated into `run_flashcard_generation()` function

### 4. Error Message Sanitization
**Issue**: Technical errors like `[Errno 21] Is a directory` shown to users.

**Check these files**:
- `gradio_interface.py`: Look for `sanitize_error_message()` function
- `gradio_interface_fixed.py`: Look for `sanitize_error_message()` function

**Expected fixes**:
- Function should convert technical errors to user-friendly messages
- Should handle common error patterns with specific explanations
- Should provide generic fallback for unknown errors
- Should be used in `run_flashcard_generation()` error handling
- Should log both original and sanitized errors for debugging

### 5. Flashcard Validation
**Issue**: Blank flashcards being created, especially audio-generated ones.

**Check these files**:
- `gradio_interface.py`: Look for `validate_flashcard()` function
- `gradio_interface_fixed.py`: Look for `validate_flashcard()` function

**Expected fixes**:
- Function should check if both question and answer are non-empty
- Should handle both Flashcard objects and dictionaries
- Should return validation status and reason
- Should be used in `flatten_flashcard_list()` function

### 6. Flashcard List Flattening
**Issue**: Nested lists and None values in flashcard lists.

**Check these files**:
- `gradio_interface.py`: Look for `flatten_flashcard_list()` function
- `gradio_interface_fixed.py`: Look for `flatten_flashcard_list()` function

**Expected fixes**:
- Function should recursively flatten nested lists
- Should filter out None values and invalid flashcards
- Should validate each flashcard before inclusion
- Should provide detailed logging of skipped items
- Should be integrated into `run_flashcard_generation()` function

### 7. Terminal Summary
**Issue**: No clear summary of generated flashcards.

**Check these files**:
- `gradio_interface.py`: Look for summary printing in `run_flashcard_generation()`
- `gradio_interface_fixed.py`: Look for summary printing in `run_flashcard_generation()`

**Expected fixes**:
- Should show total flashcards generated
- Should show breakdown by type (Text, Image Occlusion, Audio)
- Should show number of slides processed
- Should show audio file used (if applicable)

### 8. Image Processing Enhancements
**Issue**: Directory paths being treated as files.

**Check these files**:
- `flashcard_generator.py`: Look for `is_image_relevant_for_occlusion()` function
- `ankigenerator/core/image_occlusion.py`: Look for `batch_generate_image_occlusion_flashcards()` function

**Expected fixes**:
- Should include `os.path.isfile()` checks
- Should include `try-except` blocks around image operations
- Should include warning messages for skipped directories

### 9. Export Debug Print Fix
**Issue**: Debug print showing `(<class 'list'>, 'N/A')`.

**Check these files**:
- `flashcard_generator.py`: Look for `export_flashcards_to_apkg()` function

**Expected fixes**:
- Should properly categorize flashcard types
- Should show meaningful type counts instead of raw type objects
- Should handle different flashcard formats (objects, dicts, etc.)

## Integration Points to Verify

### 1. Gradio Interface Integration
**Check `run_flashcard_generation()` function in both files**:
- Should use `find_audio_file()` for audio path resolution
- Should use `flatten_flashcard_list()` after flashcard generation
- Should use `sanitize_error_message()` for error handling
- Should include terminal summary printing
- Should handle image occlusion integration properly

### 2. Image Occlusion Integration
**Check image occlusion function calls**:
- Should flatten `relevant_images` list before passing to `batch_generate_image_occlusion_flashcards()`
- Should validate and flatten occlusion flashcards before extending main list
- Should pass all required arguments to `filter_relevant_images_for_occlusion()`

### 3. Error Handling Flow
**Verify error handling chain**:
- Raw exceptions should be caught and sanitized
- Both original and user-friendly errors should be logged
- UI should display sanitized errors only
- Terminal should show full technical errors for debugging

## Test Files to Verify

### 1. `test_fixes.py`
**Check for comprehensive test coverage**:
- Should test `find_audio_file()` with various path types
- Should test `sanitize_error_message()` with different error patterns
- Should test `validate_flashcard()` with valid and invalid cards
- Should test `flatten_flashcard_list()` with nested lists and None values
- Should test export debug print functionality

### 2. `debug_flashcard_generation.py`
**Check for debugging capabilities**:
- Should test `parse_flashcards()` function
- Should test flashcard validation
- Should test list flattening
- Should test basic generation (with mock API key)

## Critical Code Patterns to Look For

### 1. Template Formatting
```python
# Should be:
enhanced_prompt = PROMPT_TEMPLATE.format(
    flashcard_type=FLASHCARD_TYPE_STRING,
    cloze=CLOZE,
    batch_text=chunk_text
)

# NOT:
enhanced_prompt = PROMPT_TEMPLATE.format(
    flashcard_type=stringify_dict(FLASHCARD_TYPE),
    # ...
)
```

### 2. Audio File Handling
```python
# Should exist:
def find_audio_file(audio_path):
    if os.path.isfile(audio_path):
        return audio_path
    elif os.path.isdir(audio_path):
        # Search for audio files
        # Return first found or None
```

### 3. Error Sanitization
```python
# Should exist:
def sanitize_error_message(error_msg):
    error_patterns = {
        "[Errno 21] Is a directory": "The system encountered a directory...",
        # ... more patterns
    }
    # Convert technical errors to user-friendly messages
```

### 4. Flashcard Validation
```python
# Should exist:
def validate_flashcard(flashcard):
    if flashcard is None:
        return False, "Flashcard is None"
    # Check question and answer are non-empty
    # Return (is_valid, reason)
```

### 5. List Flattening
```python
# Should exist:
def flatten_flashcard_list(flashcards):
    # Recursively flatten nested lists
    # Filter out None values
    # Validate each flashcard
    # Log skipped items
```

## Verification Checklist

- [ ] `get_flashcard_type_string()` function exists and converts dictionary to string
- [ ] All `PROMPT_TEMPLATE.format()` calls use `FLASHCARD_TYPE_STRING`
- [ ] `generate_multimodal_flashcards_http()` actually calls OpenAI API
- [ ] `find_audio_file()` function exists in both Gradio files
- [ ] `sanitize_error_message()` function exists in both Gradio files
- [ ] `validate_flashcard()` function exists in both Gradio files
- [ ] `flatten_flashcard_list()` function exists in both Gradio files
- [ ] Terminal summary printing exists in `run_flashcard_generation()`
- [ ] Image processing functions include file existence checks
- [ ] Export debug print shows meaningful information
- [ ] All functions are properly integrated into the main workflow
- [ ] Test files exist and cover all functionality
- [ ] Error handling chain is complete and user-friendly

## Expected Outcomes

After verification, the application should:
1. **No longer show "expected string or bytes-like object, got 'dict'" errors**
2. **Generate valid flashcards instead of "No valid flashcards"**
3. **Show user-friendly error messages instead of technical jargon**
4. **Handle audio files in directories properly**
5. **Provide clear terminal summaries**
6. **Log detailed debugging information for developers**

Please verify each item in the checklist and ensure all integration points are correctly implemented. 