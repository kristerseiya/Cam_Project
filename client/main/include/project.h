
#include <esp_camera.h>
#include <esp_log.h>

void connect2wifi(void);
void init_http(void);
void http_request_post(camera_fb_t* image_data);
esp_err_t init_camera(void);
void init_led(void);
