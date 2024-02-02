// Dimensions
const afterImage = document.getElementById('after-image');
const beforeImage = document.getElementById('before-image');
const afterDimensions = document.getElementById('after-image-dimensions');
const beforeDimensions = document.getElementById('before-image-dimensions');

afterImage.onload = function() {
    afterDimensions.textContent = `Image Size = ${afterImage.width}x${afterImage.height}`;
};

beforeImage.onload = function() {
    beforeDimensions.textContent = `Image Size = ${beforeImage.width}x${beforeImage.height}`;
};

// Sidebar
$(".menu > ul > li").click(function (e) {
    //remove active from already active
    $(this).siblings().removeClass("active");
    // add active to clicked
    $(this).toggleClass("active");
    // if has sub menu open it
    $(this).find("ul").slideToggle();
    //close other sub menu any open
    $(this).siblings().find("ul").slideUp();
    //remove active class of sub menu items
    $(this).siblings().find("ul").find("li").removeClass("active");
});