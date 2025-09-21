# Starting-with-Neural-Networks
Experimental neural-network playground for rapid deep-learning research and model training. MIT-licensed.
# 🧠 NeuralNet Playground

An experimental neural-network project for exploring deep-learning ideas, training pipelines, and custom model architectures.  
Built with stress and released under the [MIT License](LICENSE).

---

## ✨ Features
- Modular design for rapid experimentation  
- Supports PyTorch/TensorFlow backends  
- Config-driven training and evaluation  
- Easy dataset integration and augmentation tools  

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+  
- pip (or other package manager)  
- (Optional) GPU with CUDA


---------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------










##  🤓🤓The Handwritten Digits Recognition program or a neural network as nerd call it



## 🎯 What This Program Does
Imagine you want to teach a computer to look at handwritten numbers (0, 1, 2, 3... up to 9) and guess what number it is. That's exactly what this program does!

---

## 📚 Section 1: Getting Ready (Imports)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
```

**Think of this like getting your school supplies ready:**
- `torch` = Our main toolkit (like your pencil case)
- `torch.nn` = Tools to build the "brain" (neural network)
- `torch.optim` = Tools to help the brain learn better (like a good teacher)
- `torchvision` = Tools to work with pictures
- `DataLoader` = Like a conveyor belt that feeds pictures to our brain
- `tqdm` = Shows a progress bar (like a loading bar in games)
- `os` = Helps us save and organize files

---

## 🖥️ Section 2: Choosing Our Computer

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

**Simple explanation:**
- This is like choosing between a regular calculator and a super-fast gaming computer
- "cuda" = Super-fast graphics card (like a gaming computer)
- "cpu" = Regular computer processor (like a normal calculator)
- We use the fastest one available!

---

## 🧠 Section 3: Building the Brain (DigitClassifier)

```python
class DigitClassifier(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 10)
```

**Think of this like building a brain with layers:**

1. **`self.flatten`** = Takes a picture (like a 28x28 puzzle) and turns it into a long line of numbers
   - Like taking apart a LEGO house and making a long snake with all the pieces

2. **`self.fc1 = nn.Linear(28*28, 256)`** = First layer of brain cells
   - Takes 784 numbers (28×28) and connects them to 256 "brain cells"
   Like having 784 pizza slices, and each slice tells its story to 256 different taste testers
   
3. **`self.relu1 = nn.ReLU()`** = An "excitement filter"
   - Only lets positive, excited signals through
   - Like only letting happy thoughts pass to the next layer

4. **`self.dropout1 = nn.Dropout(dropout_rate)`** = A "sleepy switch"
   - Randomly makes some brain cells take a nap during learning
   - Like randomly telling 20% of students to close their eyes during a lesson
   - This prevents the brain from cheating by memorizing everything!

5. **`self.fc2 = nn.Linear(256, 128)`** = Second layer (256 → 128 brain cells)
6. **`self.fc3 = nn.Linear(128, 10)`** = Final layer (128 → 10 answers)
   - 10 because we have 10 possible answers (digits 0-9)

### How the Brain Thinks (Forward Function)

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.flatten(x)           # Turn picture into a line
    x = self.relu1(self.fc1(x))   # First layer thinks + gets excited
    x = self.dropout1(x)          # Some brain cells take a nap
    x = self.relu2(self.fc2(x))   # Second layer thinks + gets excited  
    x = self.dropout2(x)          # More brain cells nap
    x = self.fc3(x)               # Final answer (10 guesses for 0-9)
    return x
```

**Simple flow:**
Picture → Line of numbers → Brain thinks → Gets excited → Some cells nap → Brain thinks more → Gets excited → More cells nap → Final 10 guesses

---

## 🔧 Section 4: Getting Pictures Ready (Data Loading)

```python
def load_data_with_validation(batch_size: int, validation_split=0.1):
```

**This function is like a cafeteria worker preparing food:**

### Making Pictures Look the Same
```python
base_transform = transforms.Compose([
    transforms.ToTensor(),                    # Turn picture into numbers
    transforms.Normalize((0.1307,), (0.3081,)) # Make all pictures look similar
])
```
- Like making sure all photos have the same brightness and contrast

### Making Training Pictures More Interesting
```python
train_transform = transforms.Compose([
    transforms.RandomRotation(10),            # Slightly rotate some pictures
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)), # Stretch/squish some
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```
- Like showing the brain slightly tilted or stretched versions of numbers
- This helps it recognize numbers even when they're not perfect!

### Splitting Pictures into Groups
```python
full_train_dataset = datasets.MNIST(...)  # Get 60,000 training pictures
test_dataset = datasets.MNIST(...)        # Get 10,000 test pictures

# Split training into practice (54,000) and checking (6,000)
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
```
- Like having flashcards for practice, flashcards for checking, and flashcards for the final test

---

## 🎓 Section 5: Teaching the Brain (Training)

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, max_epochs=50):
```

**This is like being a teacher helping a student learn:**

### Setting Up the Classroom
```python
model.to(device)          # Put the brain on our best computer
best_val_acc = 0          # Keep track of the best score
patience_counter = 0      # Count how many times student doesn't improve
patience = 5              # Give up if no improvement for 5 tries
```

### The Learning Loop (Each Epoch = One Full Practice Session)
```python
for epoch in range(max_epochs):  # Try up to 50 practice sessions
    model.train()                # Put brain in "learning mode"
```

### Practice Time (Training Phase)
```python
for imgs, labels in train_pbar:  # Look at each picture and its answer
    imgs, labels = imgs.to(device), labels.to(device)  # Put on fast computer
    
    optimizer.zero_grad()   # Forget previous mistakes
    logits = model(imgs)    # Brain makes guesses
    loss = criterion(logits, labels)  # Calculate how wrong the guesses are
    loss.backward()         # Figure out how to fix mistakes
    optimizer.step()        # Actually fix the brain
```

**Step by step:**
1. Show the brain a picture
2. Brain makes a guess
3. Check if guess is right or wrong
4. If wrong, teach brain how to do better
5. Repeat with next picture

### Pop Quiz Time (Validation)
```python
val_accuracy = evaluate_model(model, val_loader, return_accuracy=True, verbose=False)
```
- After each practice session, give a pop quiz with new pictures
- This checks if the brain actually learned or just memorized

### Early Stopping (Smart Teacher Logic)
```python
if val_accuracy > best_val_acc:  # If this is the best score ever
    best_val_acc = val_accuracy   # Remember it
    patience_counter = 0          # Reset patience
    torch.save(model.state_dict(), 'best_mnist_model.pth')  # Save this smart brain
else:
    patience_counter += 1         # Student didn't improve
    
if patience_counter >= patience:  # If student hasn't improved 5 times in a row
    print("Early stopping!")     # Stop trying (prevents overlearning)
    break
```

---

## 📊 Section 6: Testing the Brain

```python
def evaluate_model(model, test_loader, return_accuracy=False, verbose=True):
    model.eval()  # Put brain in "test mode" (no more learning)
    
    correct, total = 0, 0
    with torch.no_grad():  # Don't change the brain during testing
        for imgs, labels in test_loader:
            logits = model(imgs)              # Brain makes guesses
            _, preds = torch.max(logits, dim=1)  # Pick the best guess
            correct += (preds == labels).sum().item()  # Count correct answers
            total += labels.size(0)           # Count total questions
```

**Like giving a final exam:**
1. Show brain pictures it's never seen before
2. Brain makes guesses
3. Count how many it got right
4. Calculate percentage score

---

## 💾 Section 7: Saving the Smart Brain

```python
def save_model(model, filepath, metadata=None):
    save_dict = {
        'model_state_dict': model.state_dict(),  # The brain's knowledge
        'model_class': model.__class__.__name__   # What type of brain it is
    }
    torch.save(save_dict, filepath)  # Save to computer file
```
- Like saving a video game - you can load the smart brain later!

---

## 🎮 Section 8: The Main Program (Putting It All Together)

```python
if __name__ == "__main__":
    # Settings
    batch_size = 64        # Look at 64 pictures at once
    learning_rate = 0.001  # How fast to learn (not too fast, not too slow)
    max_epochs = 50        # Maximum 50 practice sessions
    dropout_rate = 0.2     # Make 20% of brain cells nap during learning
    use_cnn = False        # Use simple brain (not the fancy picture brain)
```

### The Learning Process:

1. **Get Pictures Ready** 📸
   ```python
   train_loader, val_loader, test_loader = load_data_with_validation(batch_size)
   ```
   - Load 60,000 practice pictures, split them up

2. **Build the Brain** 🧠
   ```python
   model = DigitClassifier(dropout_rate=dropout_rate)
   ```
   - Create our smart brain

3. **Set Up Teacher Tools** 👨‍🏫
   ```python
   criterion = nn.CrossEntropyLoss()  # Way to measure mistakes
   optimizer = optim.Adam(...)        # Teaching method
   scheduler = optim.lr_scheduler.StepLR(...)  # Adjust learning speed
   ```

4. **Teach the Brain** 🎓
   ```python
   train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, max_epochs)
   ```
   - Practice, practice, practice!

5. **Final Test** 📝
   ```python
   test_acc = evaluate_model(model, test_loader)
   ```
   - See how well it does on brand new pictures

6. **Save the Smart Brain** 💾
   ```python
   save_model(model, 'final_mnist_model.pth', metadata)
   ```
   - Save it so we can use it later!

---

## 🎯 The Big Picture

**What happens when you run this program:**

1. 🖥️ Computer says "I'm ready to learn!"
2. 📸 Loads 70,000 pictures of handwritten numbers
3. 🧠 Builds a brain with layers that can think about pictures
4. 📚 Splits pictures into practice set, checking set, and final test set
5. 🎓 Teaches the brain by showing it thousands of examples
6. ✅ Checks how well it's learning with pop quizzes
7. 🛑 Stops when it's learned enough (or tried 50 times)
8. 📝 Gives it a final test with pictures it's never seen
9. 💾 Saves the smart brain to use later
10. 🎉 Prints the final grade (usually 98-99% correct!)

**The coolest part:** After training, you can show this brain ANY handwritten digit, and it will guess what number it is with amazing accuracy! It's like teaching a computer to see and understand numbers just like humans do! 🤖✨
