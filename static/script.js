// ---------------------------
// DROPLET ANIMATION (existing)
// ---------------------------
const dropletContainer = document.querySelector(".droplet-container");

for (let i = 0; i < 40; i++) {
    let drop = document.createElement("div");
    drop.classList.add("droplet");

    let size = Math.random() * 6 + 6;
    drop.style.width = size + "px";
    drop.style.height = size * 2 + "px";

    drop.style.left = Math.random() * 100 + "vw";
    drop.style.animationDuration = (Math.random() * 3 + 3) + "s";
    drop.style.opacity = Math.random() * 0.8 + 0.2;

    dropletContainer.appendChild(drop);
}


// ---------------------------
// POPUP FUNCTIONS (IMPORTANT)
// ---------------------------
function openModal(id) {
    document.getElementById(id).style.display = "flex";
}

function closeModal(id) {
    document.getElementById(id).style.display = "none";
}
