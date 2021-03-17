
#include "VL53L0X.h"
#include "esp_log.h"
extern "C" {
  #include "project.h"
}

/* config */
#define I2C_PORT  I2C_NUM_0
#define PIN_SDA GPIO_NUM_12
#define PIN_SCL GPIO_NUM_13

static const char *TAG = "app_main";

extern "C" void app_main()
{
    VL53L0X vl(I2C_PORT);
    vl.i2cMasterInit(PIN_SDA, PIN_SCL);
    if (!vl.init()) {
      ESP_LOGE(TAG, "Failed to initialize VL53L0X :(");
      vTaskDelay(portMAX_DELAY);
    }

    connect2wifi();
    init_http();
    init_camera();
    init_led();

    while (1) {
      /* measurement */
      uint16_t result_mm = 0;
      TickType_t tick_start = xTaskGetTickCount();
      bool res = vl.read(&result_mm);
      TickType_t tick_end = xTaskGetTickCount();
      int took_ms = ((int)tick_end - tick_start) / portTICK_PERIOD_MS;
      if (res) {
        ESP_LOGI(TAG, "Range: %d [mm] took %d [ms]", (int)result_mm, took_ms);
        if (result_mm < 100) {
          ESP_LOGI(TAG, "Taking picture...");
          camera_fb_t *pic = esp_camera_fb_get();

          // use pic->buf to access the image
          ESP_LOGI(TAG, "Picture taken! Its size was: %zu bytes", pic->len);
          http_request_post(pic);
          vTaskDelay(800 / portTICK_RATE_MS);
          }
      } else {
        ESP_LOGE(TAG, "Failed to measure :(");
      }
    }
    //
    // while (1)
    // {
    //     ESP_LOGI(TAG, "Taking picture...");
    //     camera_fb_t *pic = esp_camera_fb_get();
    //
    //     // use pic->buf to access the image
    //     ESP_LOGI(TAG, "Picture taken! Its size was: %zu bytes", pic->len);
    //     http_request_post(pic);
    //     vTaskDelay(1000 / portTICK_RATE_MS);
    // }

    // while(1) {
    //     /* Blink off (output low) */
	  //      printf("Turning off the LED\n");
    //     gpio_set_level(CONFIG_LED_LEDC_PIN, 0);
    //     vTaskDelay(100 / portTICK_PERIOD_MS);
    //     /* Blink on (output high) */
	  //      printf("Turning on the LED\n");
    //     gpio_set_level(CONFIG_LED_LEDC_PIN, 1);
    //     vTaskDelay(100 / portTICK_PERIOD_MS);
    // }
}
