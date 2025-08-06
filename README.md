# Enhanced EOS Floating Tool

A comprehensive web-based floating EOS (End of String) marker tool with advanced customization features, animations, and interactive controls.

## Features

### 🎯 Core Functionality
- **Floating EOS Display**: Interactive floating marker that can be positioned anywhere on screen
- **Real-time Updates**: Instant response to user interactions and settings changes
- **Multi-marker Support**: Create and manage multiple EOS markers simultaneously

### 🎨 Customization Options
- **Color Picker**: Choose any color for your EOS markers
- **Size Adjustment**: Resize markers from 12px to 72px
- **Font Selection**: Multiple font options including Inter, Fira Code, Roboto Mono, Arial, and Times New Roman
- **Transparency Control**: Adjust opacity from 10% to 100%

### 📍 Position and Movement
- **Draggable Interface**: Click and drag markers to any position
- **Position Memory**: Automatically remembers marker positions across page reloads
- **Snap to Edges**: Optional automatic snapping to screen edges
- **Smart Positioning**: Intelligent positioning to avoid content overlap
- **Center Button**: Quick action to center markers on screen

### ✨ Animation and Effects
- **Fade Effects**: Smooth fade in/out when showing/hiding markers
- **Bounce Animation**: Gentle bounce effect when moving markers
- **Pulse Effect**: Subtle pulsing animation option
- **Hover Effects**: Interactive scaling and shadow effects on hover

### 🛠️ Functionality Enhancements
- **Toggle Visibility**: Show/hide all markers with keyboard shortcut (Alt+V)
- **Text Editing**: Double-click markers to edit text inline
- **Export/Import Settings**: Save and load your preferences as JSON files
- **Reset Option**: Quick reset to default settings

### 🎮 User Interface
- **Settings Panel**: Comprehensive slide-out settings menu
- **Context Menu**: Right-click for quick actions (edit, clone, center, remove)
- **Keyboard Shortcuts**: 
  - `Alt + V` - Toggle visibility
  - `Alt + S` - Open settings
  - `Alt + H` - Hide controls
  - `Escape` - Close dialogs
  - `F1` - Show/hide help
- **Control Bar**: Quick access buttons for common actions

### 📱 Mobile Support
- **Responsive Design**: Optimized for mobile and tablet devices
- **Touch Controls**: Touch-friendly interface for mobile users
- **Adaptive Layout**: Settings panel adjusts to screen size

### ♿ Accessibility
- **ARIA Labels**: Proper accessibility labeling
- **Keyboard Navigation**: Full keyboard support
- **High Contrast**: Supports high contrast mode
- **Reduced Motion**: Respects user's motion preferences

### 🔧 Technical Features
- **Local Storage**: Persistent settings across browser sessions
- **Cross-browser Support**: Works on all modern browsers
- **Performance Optimized**: Efficient rendering with minimal resource usage
- **Dark Mode**: Automatic dark mode support based on system preference

## Quick Start

1. **Open the Tool**: Open `index.html` in any modern web browser
2. **Move Markers**: Click and drag the EOS marker to position it anywhere on screen
3. **Access Settings**: Click the gear icon (⚙️) or press `Alt + S` to open settings
4. **Customize**: Adjust colors, size, font, and animation preferences
5. **Add Markers**: Use the "Add New Marker" button to create multiple markers
6. **Save Settings**: Your preferences are automatically saved

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Alt + V` | Toggle marker visibility |
| `Alt + S` | Open/close settings panel |
| `Alt + H` | Hide/show control bar |
| `Escape` | Close all open panels |
| `F1` | Show/hide help overlay |
| `Enter` | Finish text editing |

## Mouse Controls

| Action | Result |
|--------|--------|
| **Drag** | Move marker to new position |
| **Right-click** | Open context menu |
| **Double-click** | Edit marker text |
| **Hover** | Show interactive effects |

## Settings Export/Import

### Export Settings
1. Open settings panel
2. Click "Export Settings"
3. Save the downloaded JSON file

### Import Settings
1. Open settings panel
2. Click "Import Settings"
3. Select your previously exported JSON file

## Browser Compatibility

- ✅ Chrome 80+
- ✅ Firefox 75+
- ✅ Safari 13+
- ✅ Edge 80+

## File Structure

```
/
├── index.html          # Main HTML structure
├── styles.css          # Complete styling and animations
├── script.js           # JavaScript functionality
├── README.md           # This documentation
└── EOS.py             # Original Python implementation (separate tool)
```

## Advanced Usage

### Multiple Markers
- Add up to 10 markers on screen simultaneously
- Each marker can have different text and positioning
- Clone existing markers with right-click context menu

### Smart Features
- **Auto-positioning**: Prevents markers from overlapping with the settings panel
- **Snap to edges**: Automatically aligns markers to screen edges when enabled
- **Responsive behavior**: Markers automatically adjust when window is resized

### Customization Tips
- Use high contrast colors for better visibility
- Enable pulse effect for attention-grabbing markers
- Adjust opacity for subtle background presence
- Choose monospace fonts (Fira Code, Roboto Mono) for code-related markers

## Troubleshooting

### Settings Not Saving
- Ensure your browser allows local storage
- Try clearing browser cache and reloading
- Check browser's privacy settings

### Performance Issues
- Disable animations in settings if experiencing lag
- Reduce number of active markers
- Close other browser tabs to free up resources

### Mobile Issues
- Use touch gestures instead of mouse actions
- Tap gear icon to access settings
- Use two-finger scroll in settings panel

## Contributing

This is a standalone web application. To contribute:

1. Fork the repository
2. Make your changes to HTML, CSS, or JavaScript files
3. Test across different browsers
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues or feature requests, please create an issue in the repository.
