use std::{
    borrow::Cow,
    ffi::{c_char, CStr},
    process::id,
    sync::Mutex,
    thread::{current, park},
};

use ash::{extensions::ext, extensions::ext::DebugUtils, extensions::khr, vk};
use gpu_allocator::{
    vulkan::{Allocator, AllocatorCreateDesc},
    AllocatorDebugSettings,
};
use log::{debug, error, info, logger, trace, warn};

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            info!("{message}");
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            warn!("{message}");
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!("🆘 {message}");
            debug!(
                "🛑 PARKING THREAD `{}` -> attach debugger to pid {}!",
                current().name().unwrap_or_default(),
                id()
            );

            logger().flush();

            park();
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            trace!("{message}");
        }
        _ => {}
    }

    vk::FALSE
}

pub struct Device {
    pub device: ash::Device,
    pub instance: ash::Instance,
    pub debug_loader: DebugUtils,
    pub debug_callback: vk::DebugUtilsMessengerEXT,
    pub allocator: Option<Mutex<gpu_allocator::vulkan::Allocator>>,
    pub queue_family_index: u32,
}
impl Device {
    pub fn create() -> Self {
        unsafe {
            let entry = ash::Entry::linked();
            let app_name = CStr::from_bytes_with_nul_unchecked(b"Test Vulkan\n");

            let layer_names = [CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            )];
            let layers_names_raw: Vec<*const c_char> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let extensions = [
                DebugUtils::name().as_ptr(),
                // khr::DeviceGroup::name().as_ptr(),
            ];

            let appinfo = vk::ApplicationInfo::builder()
                .application_name(app_name)
                .application_version(0)
                .engine_name(app_name)
                .engine_version(0)
                .api_version(vk::API_VERSION_1_2);

            let create_flags = vk::InstanceCreateFlags::default();

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&appinfo)
                .enabled_layer_names(&layers_names_raw)
                .enabled_extension_names(&extensions)
                .flags(create_flags);

            let instance = entry
                .create_instance(&create_info, None)
                .expect("Could not Create Instance!");

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_loader = DebugUtils::new(&entry, &instance);
            let debug_callback = debug_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap();

            let pdevice = instance
                .enumerate_physical_devices()
                .expect("No Physical Device found!");

            let (pdevice, queue_family_index) = pdevice
                .iter()
                .filter_map(|pdevice| {
                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| Some((*pdevice, index)))
                })
                .max_by_key(|(pdevice, info)| {
                    let ty = instance
                        .get_physical_device_properties(*pdevice)
                        .device_type;
                    match ty {
                        vk::PhysicalDeviceType::CPU => 1,
                        vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                        vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                        _ => 0,
                    }
                })
                .unwrap();

            let properties = instance.get_physical_device_properties(pdevice);
            trace!(
                "Found Physical Device: {:?}",
                CStr::from_bytes_until_nul(std::mem::transmute(properties.device_name.as_ref()))
                    .unwrap()
            );

            let queue_family_index = queue_family_index as u32;

            let device_extension = [khr::BufferDeviceAddress::name().as_ptr()];

            let properties = [1.0];

            let queue_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&properties);

            let fp_v1_1 = instance.fp_v1_1();
            let get_physical_device_features2 = fp_v1_1.get_physical_device_features2;

            let mut vulkan_1_1_features = vk::PhysicalDeviceVulkan11Features::builder();
            let mut vulkan_1_2_features =
                vk::PhysicalDeviceVulkan12Features::builder().buffer_device_address(true);

            let features2 = vk::PhysicalDeviceFeatures2::builder()
                .push_next(&mut vulkan_1_1_features)
                .push_next(&mut vulkan_1_2_features);

            let mut features2 = features2.build();

            get_physical_device_features2(pdevice, &mut features2);

            // Testing if required features are enabled
            if vulkan_1_2_features.buffer_device_address != vk::TRUE {
                log::error!("BufferDeviceAddress could not be enabled!");
                panic!();
            }

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&device_extension)
                .push_next(&mut features2);

            let device = instance
                .create_device(pdevice, &device_create_info, None)
                .unwrap();

            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: pdevice,
                debug_settings: AllocatorDebugSettings {
                    log_leaks_on_shutdown: true,
                    log_memory_information: true,
                    log_allocations: true,
                    ..Default::default()
                },
                buffer_device_address: true,
            })
            .unwrap();
            let allocator = Some(Mutex::new(allocator));
            Self {
                device,
                instance,
                debug_loader,
                debug_callback,
                allocator,
                queue_family_index,
            }
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.allocator.take();
            self.device.destroy_device(None);
            self.debug_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
            trace!("Dropped Device.");
        }
    }
}
