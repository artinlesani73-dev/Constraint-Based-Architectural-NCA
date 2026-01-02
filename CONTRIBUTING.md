# Contributing to Constraint-Based Architectural NCA

Thank you for your interest in contributing to this project!

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker
- Include a clear description of the problem
- Provide steps to reproduce if applicable
- Include system information (OS, Python version, PyTorch version)

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Test your changes
5. Commit with clear messages (`git commit -m 'Add feature X'`)
6. Push to your fork (`git push origin feature/your-feature`)
7. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add comments for complex logic
- Update documentation if needed

### Areas for Contribution

- **New constraints** - Implement additional architectural constraints
- **UI improvements** - Enhance the web interface
- **Performance** - Optimize training or inference
- **Documentation** - Improve docs, add examples
- **Fine-tuning recipes** - New fine-tuning configurations for different aesthetics

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Constraint-Based-Architectural-NCA.git
cd Constraint-Based-Architectural-NCA

# Install dependencies
pip install -r deploy/requirements.txt

# Run tests
python deploy/test_model.py

# Start the dev server
python deploy/server.py
```

## Questions?

Feel free to open an issue for any questions about contributing.
