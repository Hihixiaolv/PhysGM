window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Check for click events on the navbar burger icon

	if (typeof AdobeDC !== 'undefined') {
        var adobeDCView = new AdobeDC.View({
            clientId: "YOUR_CLIENT_ID_HERE", // 请替换为您的Adobe DC View API客户端ID
            divId: "adobe-dc-view"
        });
        
        // Load PDF document
        adobeDCView.previewFile({
            content: {
                location: {
                    url: "static/pdfs/pipeline3.pdf" // 请替换为您的PDF文件路径
                }
            },
            metaData: {
                fileName: "PhysGM_Paper.pdf",
                fileSize: "1 MB"
            }
        }, {
            embedMode: "SIZED_CONTAINER",
            showDownloadPDF: true,
            showPrintPDF: true
        });
    }
	
    // Tab switching functionality
    $('.tabs li').click(function() {
        var tabId = $(this).attr('data-tab');
        
        // Remove active class from all tabs and content
        $('.tabs li').removeClass('is-active');
        $('.tab-pane').removeClass('is-active');
        
        // Add active class to clicked tab and corresponding content
        $(this).addClass('is-active');
        $('#' + tabId).addClass('is-active');
    });
	
    bulmaSlider.attach();
})

// Video Carousel Functionality
$(document).ready(function() {
    const carousel = $('#videoCarousel');
    const dots = $('.dot');
    const prevBtn = $('#prevBtn');
    const nextBtn = $('#nextBtn');
    let isTransitioning = false;

    // 可见卡片数（动态计算）
    const $cards = carousel.children('.video-card');
    const totalSlides = $cards.length;
    const visibleCount = Math.max(3, totalSlides); 

    
    // 克隆前后卡片
    for (let i = 0; i < visibleCount; i++) {
        carousel.append($cards.eq(i).clone());
        carousel.prepend($cards.eq(totalSlides - 1 - i).clone());
    }
    // 重新获取所有卡片
    const $allCards = carousel.children('.video-card');
    let currentIndex = visibleCount; // 初始显示第一个真实卡片

    function setCarouselPosition(index, animate = true) {
        const containerWidth = $('.video-carousel-container').width();
        // 计算当前卡片左边所有卡片的宽度
        let leftWidth = 0;
        for (let i = 0; i < index; i++) {
            leftWidth += $allCards.eq(i).outerWidth(true);
        }
        // 当前卡片宽度
        const cardWidth = $allCards.eq(index).outerWidth(true);
        // 让当前卡片中心对齐容器中心
        const translateX = containerWidth / 2 - leftWidth - cardWidth / 2;
        if (animate) {
            carousel.css('transition', 'transform 0.5s cubic-bezier(0.4,0,0.2,1)');
        } else {
            carousel.css('transition', 'none');
        }
        carousel.css('transform', `translateX(${translateX}px)`);
        // 激活卡片
        $allCards.removeClass('active');
        $allCards.eq(index).addClass('active');
        // 激活圆点
        dots.removeClass('active');
        dots.eq((index - visibleCount + totalSlides) % totalSlides).addClass('active');
    }

    setCarouselPosition(currentIndex, false);

    function goTo(index) {
        if (isTransitioning) return;
        isTransitioning = true;
        setCarouselPosition(index, true);
        currentIndex = index;
    }

    // 修复无限循环逻辑
    carousel.on('transitionend webkitTransitionEnd', function() {
        // 如果到达了克隆的最后一个视频，跳转到真实的第一个视频
        if (currentIndex >= totalSlides + visibleCount) {
            currentIndex = visibleCount;
            setCarouselPosition(currentIndex, false);
        }
        // 如果到达了克隆的第一个视频，跳转到真实的最后一个视频
        else if (currentIndex < visibleCount) {
            currentIndex = totalSlides + visibleCount - 1;
            setCarouselPosition(currentIndex, false);
        }
        isTransitioning = false;
    });

    nextBtn.click(function() {
        if (isTransitioning) return;
        goTo(currentIndex + 1);
    });
    prevBtn.click(function() {
        if (isTransitioning) return;
        goTo(currentIndex - 1);
    });
    dots.click(function() {
        if (isTransitioning) return;
        const index = $(this).data('index');
        goTo(index + visibleCount);
    });

    // 自动播放
    let autoPlayInterval;
    function startAutoPlay() {
        autoPlayInterval = setInterval(function() {
            goTo(currentIndex + 1);
        }, 5000);
    }
    function stopAutoPlay() {
        if (autoPlayInterval) clearInterval(autoPlayInterval);
    }
    startAutoPlay();
    carousel.hover(stopAutoPlay, startAutoPlay);
    prevBtn.hover(stopAutoPlay, startAutoPlay);
    nextBtn.hover(stopAutoPlay, startAutoPlay);

    // 键盘
    $(document).keydown(function(e) {
        if (e.keyCode === 37) prevBtn.click();
        else if (e.keyCode === 39) nextBtn.click();
    });

    // 触摸滑动
    let startX = 0, endX = 0;
    carousel.on('touchstart', function(e) {
        startX = e.originalEvent.touches[0].clientX;
    });
    carousel.on('touchend', function(e) {
        endX = e.originalEvent.changedTouches[0].clientX;
        handleSwipe();
    });
    function handleSwipe() {
        const swipeThreshold = 50;
        const diff = startX - endX;
        if (Math.abs(diff) > swipeThreshold) {
            if (diff > 0) nextBtn.click();
            else prevBtn.click();
        }
    }

    // 窗口大小变化
    $(window).resize(function() {
        setCarouselPosition(currentIndex, false);
    });
});

// Video Grid Functionality
$(document).ready(function() {
    // 创建模态框
    const modal = $('<div class="video-modal">' +
        '<div class="modal-content">' +
        '<span class="modal-close">&times;</span>' +
        '<video autoplay muted loop playsinline>' +
        '</video>' +
        '<div class="modal-info">' +
        '<h3 class="modal-title"></h3>' +
        '<p class="modal-description"></p>' +
        '</div>' +
        '</div>' +
        '</div>');
    
    $('body').append(modal);
    
    // 点击视频网格项打开模态框
    $('.grid-video-item').click(function() {
        const video = $(this).find('video');
        const title = $(this).find('.video-title').text();
        const description = $(this).find('.video-description').text();
        
        // 设置模态框内容
        modal.find('video').attr('src', video.attr('src'));
        modal.find('.modal-title').text(title);
        modal.find('.modal-description').text(description);
        
        // 显示模态框
        modal.addClass('active');
        $('body').addClass('modal-open');
    });
    
    // 关闭模态框
    modal.find('.modal-close').click(function() {
        modal.removeClass('active');
        $('body').removeClass('modal-open');
        modal.find('video')[0].pause();
    });
    
    // 点击模态框背景关闭
    modal.click(function(e) {
        if (e.target === this) {
            modal.removeClass('active');
            $('body').removeClass('modal-open');
            modal.find('video')[0].pause();
        }
    });
    
    // ESC键关闭模态框
    $(document).keydown(function(e) {
        if (e.keyCode === 27 && modal.hasClass('active')) {
            modal.removeClass('active');
            $('body').removeClass('modal-open');
            modal.find('video')[0].pause();
        }
    });
});
