use crate::TextLine;

/// A single recognition event emitted during streaming OCR.
#[derive(Debug, Clone)]
pub enum OcrEvent {
    /// A text line has been recognized.
    LineRecognized(TextLine),
    /// Page preprocessing is complete, with the number of lines detected.
    LayoutDetected { line_count: usize },
    /// Processing is complete.
    Done,
}

/// Configuration for streaming OCR pipeline.
/// This is a marker trait for types that can produce OcrEvent streams.
pub trait OcrStream {
    /// Get the next OCR event. Returns None when processing is complete.
    fn next_event(&mut self) -> Option<OcrEvent>;
}

/// Adapter to convert OcrStream into a standard Iterator over TextLines.
pub struct TextLineIterator<S: OcrStream> {
    stream: S,
}

impl<S: OcrStream> TextLineIterator<S> {
    pub fn new(stream: S) -> Self {
        Self { stream }
    }
}

impl<S: OcrStream> Iterator for TextLineIterator<S> {
    type Item = TextLine;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.stream.next_event()? {
                OcrEvent::LineRecognized(line) => return Some(line),
                OcrEvent::LayoutDetected { .. } => continue,
                OcrEvent::Done => return None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BBox;

    struct MockStream {
        events: Vec<OcrEvent>,
        idx: usize,
    }

    impl MockStream {
        fn new(events: Vec<OcrEvent>) -> Self {
            Self { events, idx: 0 }
        }
    }

    impl OcrStream for MockStream {
        fn next_event(&mut self) -> Option<OcrEvent> {
            if self.idx < self.events.len() {
                let event = self.events[self.idx].clone();
                self.idx += 1;
                Some(event)
            } else {
                None
            }
        }
    }

    #[test]
    fn test_text_line_iterator() {
        let events = vec![
            OcrEvent::LayoutDetected { line_count: 2 },
            OcrEvent::LineRecognized(TextLine {
                text: "hello".to_string(),
                bbox: BBox::new(0, 0, 100, 20),
                confidence: 0.9,
            }),
            OcrEvent::LineRecognized(TextLine {
                text: "world".to_string(),
                bbox: BBox::new(0, 30, 100, 20),
                confidence: 0.85,
            }),
            OcrEvent::Done,
        ];

        let stream = MockStream::new(events);
        let lines: Vec<TextLine> = TextLineIterator::new(stream).collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "hello");
        assert_eq!(lines[1].text, "world");
    }

    #[test]
    fn test_empty_stream() {
        let events = vec![OcrEvent::LayoutDetected { line_count: 0 }, OcrEvent::Done];
        let stream = MockStream::new(events);
        let lines: Vec<TextLine> = TextLineIterator::new(stream).collect();
        assert!(lines.is_empty());
    }

    #[test]
    fn test_stream_exhaustion() {
        let stream = MockStream::new(vec![]);
        let lines: Vec<TextLine> = TextLineIterator::new(stream).collect();
        assert!(lines.is_empty());
    }
}
