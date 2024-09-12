window.addEventListener('scroll', function() {
    var sections = document.querySelectorAll('.pages:not(#try)'); // Exclude the 'Try' page
    var navLinks = document.querySelectorAll('#navigation-right a');
    var scrollPosition = window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0;

    sections.forEach(function(section, index) {
        var top = section.offsetTop - 100;
        var bottom = top + section.offsetHeight;

        // Check if the current scroll position is within the bounds of the current section
        if (scrollPosition >= top && scrollPosition < bottom) {
            // Remove 'active' class from all navigation links
            navLinks.forEach(function(navLink) {
                navLink.classList.remove('active');
            });
            // Add 'active' class to the corresponding navigation link
            navLinks[index].classList.add('active');
        }
    });
});

// for the features
// Query selectors for details and buttons
document.addEventListener("DOMContentLoaded", function() {
var gsaptime = gsap.timeline();
gsaptime.from("#navigation-bar",{
    y : -100,
    duration : 0.3,
    opacity : 0,
})

gsaptime.from("#navigation-right a",{
    y : -100,
    duration : 0.2,
    stagger :0.05,
    opacity : 0,
})

gsaptime.from("#home-heading ",{
    y : -100,
    duration : 0.5,
    opacity : 0,
})

var details = document.querySelectorAll(".details");
var buttons = document.querySelectorAll(".btn");
var minusicon = '<i class="ri-subtract-line"></i>';
var plusicon = '<i class="ri-add-line"></i>';
var h3s = []; // Array to store the names of the buttons

// Initially display first details and set minus icon for the first button
details[0].style.display = "block";

// Store the names of the buttons and initialize button content with icons
buttons.forEach(function(btn, index) {
    h3s.push(btn.innerHTML); // Store the initial content of the button (h3 tag)
    if (index === 0) {
        btn.innerHTML = h3s[index] + minusicon; // Set the button content to include the minus icon for the first button
    } else {
        btn.innerHTML = h3s[index] + plusicon; // Set the button content to include the plus icon for other buttons
    }
});

// Add event listener to each button
buttons.forEach(function(eachbtn, index) {
    eachbtn.addEventListener('click', function() {
        disappearAll();
        // Toggle details display
        if (details[index].style.display === "block") {
            details[index].style.display = "none";
            eachbtn.innerHTML = h3s[index] + plusicon; // Set the button content to include the plus icon
        } else {
            details[index].style.display = "block";
            eachbtn.innerHTML = h3s[index] + minusicon; // Set the button content to include the minus icon
        }
        
        // Reset all other buttons to plus icon
        buttons.forEach(function(btn, idx) {
            if (idx !== index) {
                btn.innerHTML = h3s[idx] + plusicon; // Restore the original button content and add the plus icon
            }
            btn.classList.remove('animate');
        });

        // Add animation class to the clicked button
        eachbtn.classList.add('animate');
    })
});

function disappearAll() {
    details.forEach(function(i) {
        // i.style.transition = "display 2s"; // Apply transition to the display property
        i.style.display = "none";
    });
}});

//feature end

// // contact form submit button code start
document.addEventListener("DOMContentLoaded", function() {
    var submitbtn = document.querySelector("#contactbtn");
    var inputs = document.querySelectorAll(".inputs");

    submitbtn.addEventListener('click', function(event) {
        var alertShown = false; // Variable to track if alert has been shown

        inputs.forEach(function(input) {
            if (input.value.trim() === "") {
                if (!alertShown) { // Only show alert if not already shown
                    alert("Please enter all the details.");
                    alertShown = true; // Set alertShown to true
                }
                event.preventDefault(); // Prevent form submission
            }
        });
    });
});
// // contact form submit button code end

// try code start
document.addEventListener("DOMContentLoaded", function() {
    // File submission checking
    var file = document.querySelector("#uploadBtn");
    file.addEventListener('change', () => {
        if (file.files.length && !file.files[0].name.includes(".csv")) {
            alert("Only CSV files are accepted!");
            file.value = "";
        }
    });

    // Submit button click handler for form validation
    var submitbtn = document.querySelector("#submit-button");
    var inputs = document.querySelectorAll(".text");

    submitbtn.addEventListener('click', function(event) {
        var alertShown = false; // Variable to track if alert has been shown

        inputs.forEach(function(input) {
            if (input.value.trim() === "") {
                if (!alertShown) { // Only show alert if not already shown
                    alert("Please enter all the details.");
                    alertShown = true; // Set alertShown to true
                }
                event.preventDefault(); // Prevent form submission
            }
        });
    });

    // File name list
    var fileInput = document.querySelector("#uploadBtn");
    var fileListContainer = document.querySelector("#file-list");

    fileInput.addEventListener('change', function() {
        // Clear previous file list
        fileListContainer.innerHTML = '';

        // Loop through each selected file
        for (var i = 0; i < fileInput.files.length; i++) {
            var file = fileInput.files[i];
            var fileItem = document.createElement('div');
            fileItem.className = 'files';

            // Create a text node with the file name
            var fileName = document.createTextNode(file.name);

            // Create a remove button
            var removeButton = document.createElement('button');
            removeButton.textContent = 'Remove';
            removeButton.className = 'remove-button';

            // Event listener to remove file item and clear file input
            removeButton.addEventListener('click', function() {
                fileInput.value = ''; // Clear the file input
                fileListContainer.innerHTML = ''; // Clear the file list container
            });

            // Append the file name and remove button to the file item
            fileItem.appendChild(fileName);
            fileItem.style.color = "black";
            fileItem.style.backgroundColor  = "#D4D9E7";
            fileItem.appendChild(removeButton);

            // Append the file item to the file list container
            fileListContainer.appendChild(fileItem);
        }
    });

    // Submit form handler
var form = document.getElementById("uploadForm");

form.addEventListener('submit', function(event) {
    const expectations = document.getElementById('expectations').value.trim();
    const fileUpload = document.getElementById('uploadBtn').files.length;
    const urlLink = document.querySelector('textarea[name="url_link"]').value.trim();

    if (!expectations || (fileUpload === 0 && !urlLink)) {
        event.preventDefault(); // Prevent form submission if expectations are not provided or neither file nor URL link is given
        alert('Please provide expectations and either upload a file or provide a URL link.');
    } else {
        event.preventDefault(); // Prevent default form submission

        // Send form data to server using fetch API
        fetch('/submit', {
            method: 'POST',
            body: new FormData(form)
        })
        .then(response => response.text()) // Parse response as text
        .then(result => {
            // Update result div with the received result
            const resultDiv = document.getElementById("result");
            const textElement = document.getElementById("text");

            if (result.trim() != '') {
                textElement.style.display = 'block'; // Show the result div if result is not empty
                textElement.innerHTML = result;
            }
            else{
                textElement.style.display = "none";
            }
        })
        .catch(error => console.error('Error:', error));
    }
});
});
