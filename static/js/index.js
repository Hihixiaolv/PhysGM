// 阻止 Video.js 的帮助改进提示
window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {

    // --- 全局功能 ---


    // Tab 切换功能
    $('.tabs li').click(function() {
        var tabId = $(this).attr('data-tab');
        $('.tabs li').removeClass('is-active');
        $('.tab-pane').removeClass('is-active');
        $(this).addClass('is-active');
        $('#' + tabId).addClass('is-active');
    });

    // Bulma Slider 初始化
    if (typeof bulmaSlider !== 'undefined') {
        bulmaSlider.attach();
    }



});