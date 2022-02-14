// Based on the Vulkano triangle example.

// Triangle example Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use png;
use std::io::Cursor;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::format::Format;
use vulkano::image::immutable::ImmutableImage;
use vulkano::image::ImageCreateFlags;
use vulkano::image::MipmapsCount;
use vulkano::image::{
    view::ImageView, ImageAccess, ImageDimensions, ImageUsage, StorageImage, SwapchainImage,
};
use vulkano::instance::Instance;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::swapchain::{self, AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::Version;
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

// We'll make our Color type an RGBA8888 pixel.
type Color = (u8, u8, u8, u8);

const WIDTH: usize = 320;
const HEIGHT: usize = 240;

type Animation = (Vec<(usize, usize)>, Vec<f32>, bool);

enum AlphaChannel {
    First,
    Last,
}
fn premultiply(img: &mut [Color], alpha: AlphaChannel) {
    match alpha {
        AlphaChannel::First => {
            for px in img.iter_mut() {
                let a = px.0 as f32 / 255.0;
                px.1 = (*px.1 as f32 * a).round() as u8;
                px.2 = (*px.2 as f32 * a).round() as u8;
                px.3 = (*px.3 as f32 * a).round() as u8;
                // swap around to rgba8888
                let a = px.0;
                px.0 = px.1;
                px.1 = px.2;
                px.2 = px.3;
                px.3 = a;
            }
        }
        AlphaChannel::Last => {
            for px in img.iter_mut() {
                let a = px.3.unwrap() as f32 / 255.0;
                px.0 = (px.0 as f32 * a) as u8;
                px.1 = (px.1 as f32 * a) as u8;
                px.2 = (px.2 as f32 * a) as u8;
                // already rgba8888
            }
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32, // Float positions and extents could also be fine
}

impl Rect {
    pub fn new(x: i32, y: i32, w: u32, h: u32) -> Rect {
        Rect { x, y, w, h }
    }
    // Maybe add functions to test if a point is inside the rect...
    // Or whether two rects overlap...
}

#[derive(PartialEq, Eq, Clone, Copy, Hash, Debug)]
pub struct Vec2i {
    // Or Vec2f for floats?
    pub x: i32,
    pub y: i32,
}

impl Vec2i {
    pub fn new(x: i32, y: i32) -> Vec2i {
        Vec2i { x, y }
    }
    // Maybe add functions for e.g. the midpoint of two vecs, or...
}

pub struct Image {
    pub buffer: Box<[Color]>, // or Vec<Color>, or...
    pub w: usize,
    pub h: usize,
}

impl Image {
    // maybe a function to load it from a file using the png crate!
    // you'd want to premultiply it and convert it from `[u8]` into `[Color]`!
    fn get_png() {
        let png_bytes = include_bytes!("../spritesheet.png").to_vec();
        let cursor = Cursor::new(png_bytes);
        let decoder = png::Decoder::new(cursor);
        let mut reader = decoder.read_info().unwrap();
        let info = reader.info();
        let dimensions = ImageDimensions::Dim2d {
            width: info.width,
            height: info.height,
            array_layers: 1,
        };
        let mut image_data = Vec::new();
        image_data.resize((info.width * info.height * 4) as usize, 0);
        reader.next_frame(&mut image_data).unwrap();

        let (image, future) = ImmutableImage::from_iter(
            image_data.iter().cloned(),
            dimensions,
            MipmapsCount::One,
            Format::R8G8B8A8_SRGB,
            queue.clone(),
        )
        .unwrap();
        (ImageView::new(image).unwrap(), future)
    }

    // maybe put bitblt into here?
    pub fn bitblt(src: &Image, from: Rect, dst: &mut Image, to: Vec2) {
        assert!(rect_inside(from, (0, 0, src_size.0, src_size.1)));
        let (to_x, to_y) = to;
        if (to_x + from.x as i32) < 0
            || (dst_size.0 as i32) <= to_x
            || (to_y + from.3 as i32) < 0
            || (dst_size.1 as i32) <= to_y
        {
            return;
        }
        let src_pitch = src_size.0;
        let dst_pitch = dst_size.0;
        // All this rigmarole is just to avoid bounds checks on each pixel of the blit.
        // We want to calculate which row/col of the src image to start at and which to end at.
        // This way there's no need to even check for out of bounds draws---
        // we'll skip rows that are off the top or off the bottom of the image
        // and skip columns off the left or right sides.
        let y_skip = to_y.max(0) - to_y;
        let x_skip = to_x.max(0) - to_x;
        let y_count = (to_y + from.3 as i32).min(dst_size.1 as i32) - to_y;
        let x_count = (to_x + from.2 as i32).min(dst_size.0 as i32) - to_x;
        // The code above is gnarly so these are just for safety:
        debug_assert!(0 <= x_skip);
        debug_assert!(0 <= y_skip);
        debug_assert!(0 <= x_count);
        debug_assert!(0 <= y_count);
        debug_assert!(x_count <= from.2 as i32);
        debug_assert!(y_count <= from.3 as i32);
        debug_assert!(0 <= to_x + x_skip);
        debug_assert!(0 <= to_y + y_skip);
        debug_assert!(0 <= from.0 as i32 + x_skip);
        debug_assert!(0 <= from.1 as i32 + y_skip);
        debug_assert!(to_x + x_count <= dst_size.0 as i32);
        debug_assert!(to_y + y_count <= dst_size.1 as i32);
        // OK, let's do some copying now
        for (row_a, row_b) in src
            // From the first pixel of the top row to the first pixel of the row past the bottom...
            [(src_pitch * (from.1 as i32 + y_skip) as usize)..(src_pitch * (from.1 as i32 + y_count) as usize)]
            // For each whole row...
            .chunks_exact(src_pitch)
            // Tie it up with the corresponding row from dst
            .zip(
                dst[(dst_pitch * (to_y + y_skip) as usize)
                    ..(dst_pitch * (to_y + y_count) as usize)]
                    .chunks_exact_mut(dst_pitch),
            )
            {
            // Get column iterators, save on indexing overhead
            let to_cols = row_b
                [((to_x + x_skip) as usize)..((to_x + x_count) as usize)].iter_mut();
            let from_cols = row_a
                [((from.0 as i32 + x_skip) as usize)..((from.0 as i32 + x_count) as usize)].iter();
            // Composite over, assume premultiplied rgba8888 in src!
            for (to, from) in to_cols.zip(from_cols) {
                let ta = to.3 as f32 / 255.0;
                let fa = from.3 as f32 / 255.0;
                to.0 = from.0.saturating_add((to.0 as f32 * (1.0 - fa)).round() as u8);
                to.1 = from.1.saturating_add((to.1 as f32 * (1.0 - fa)).round() as u8);
                to.2 = from.2.saturating_add((to.2 as f32 * (1.0 - fa)).round() as u8);
                to.3 = ((fa + ta * (1.0 - fa)) * 255.0).round() as u8;
            }
        }
    }

    // ... what else?

    // ... Could our 2d framebuffer itself be an `Image`?
}

#[derive(Copy, Clone)]
struct Rect2 {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    color: Color,
}

impl Rect2 {
    fn change_color_to(&mut self, new: Color) {
        self.color = new;
    }

    fn left(self) -> usize {
        self.x
    }

    fn right(self) -> usize {
        self.x + self.width
    }

    fn top(self) -> usize {
        self.y
    }

    fn bottom(self) -> usize {
        self.y + self.height
    }

    fn change_x_by(&mut self, change: f32) {
        let mut x = self.x as f32;
        x = x + change;

        if x < 0.0 {
            self.x = 0;
        } else if x as usize + self.width > WIDTH {
            self.x = WIDTH - self.width;
        } else {
            self.x = x as usize;
        }
    }

    fn change_y_by(&mut self, change: f32) {
        let mut y = self.y as f32;
        y = y + change;

        if y < 0.0 {
            self.y = 0;
        } else if y as usize + self.height > HEIGHT {
            self.y = HEIGHT - self.height;
        } else {
            self.y = y as usize;
        }
    }
}

// Here's what clear looks like, though we won't use it
#[allow(dead_code)]
fn clear(fb: &mut [Color], c: Color) {
    fb.fill(c);
}

#[allow(dead_code)]
fn draw_filled_rect(fb: &mut [Color], rect: Rect) {
    for y in rect.top()..rect.bottom() {
        fb[(y * WIDTH + rect.left())..(y * WIDTH + rect.right())].fill(rect.color);
    }
}

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, Version::V1_1, &required_extensions, None).unwrap();
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();
    let (device, mut queues) = Device::new(
        physical_device,
        &Features::none(),
        &physical_device
            .required_extensions()
            .union(&device_extensions),
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();
    let queue = queues.next().unwrap();
    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical_device).unwrap();
        let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        Swapchain::start(device.clone(), surface.clone())
            .num_images(caps.min_image_count)
            .format(format)
            .dimensions(dimensions)
            .usage(ImageUsage::color_attachment())
            .sharing_mode(&queue)
            .composite_alpha(composite_alpha)
            .build()
            .unwrap()
    };

    // We now create a buffer that will store the shape of our triangle.
    #[derive(Default, Debug, Clone)]
    struct Vertex {
        position: [f32; 2],
        uv: [f32; 2],
    }
    vulkano::impl_vertex!(Vertex, position, uv);

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        [
            Vertex {
                position: [-1.0, -1.0],
                uv: [0.0, 0.0],
            },
            Vertex {
                position: [3.0, -1.0],
                uv: [2.0, 0.0],
            },
            Vertex {
                position: [-1.0, 3.0],
                uv: [0.0, 2.0],
            },
        ]
        .iter()
        .cloned(),
    )
    .unwrap();
    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
                #version 450

                layout(location = 0) in vec2 position;
                layout(location = 1) in vec2 uv;
                layout(location = 0) out vec2 out_uv;
                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                    out_uv = uv;
                }
            "
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
                #version 450

                layout(set = 0, binding = 0) uniform sampler2D tex;
                layout(location = 0) in vec2 uv;
                layout(location = 0) out vec4 f_color;

                void main() {
                    f_color = texture(tex, uv);
                }
            "
        }
    }

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    // Here's our (2D drawing) framebuffer.
    let mut fb2d = [(128, 64, 64, 255); WIDTH * HEIGHT];
    // We'll work on it locally, and copy it to a GPU buffer every frame.
    // Then on the GPU, we'll copy it into an Image.
    let fb2d_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::transfer_source(),
        false,
        (0..WIDTH * HEIGHT).map(|_| (255_u8, 0_u8, 0_u8, 0_u8)),
    )
    .unwrap();
    // Let's set up the Image we'll copy into:
    let dimensions = ImageDimensions::Dim2d {
        width: WIDTH as u32,
        height: HEIGHT as u32,
        array_layers: 1,
    };
    let fb2d_image = StorageImage::with_usage(
        device.clone(),
        dimensions,
        Format::R8G8B8A8_UNORM,
        ImageUsage {
            // This part is key!
            transfer_destination: true,
            sampled: true,
            storage: true,
            transfer_source: false,
            color_attachment: false,
            depth_stencil_attachment: false,
            transient_attachment: false,
            input_attachment: false,
        },
        ImageCreateFlags::default(),
        std::iter::once(queue_family),
    )
    .unwrap();
    // Get a view on it to use as a texture:
    let fb2d_texture = ImageView::new(fb2d_image.clone()).unwrap();

    let fb2d_sampler = Sampler::new(
        device.clone(),
        Filter::Linear,
        Filter::Linear,
        MipmapMode::Nearest,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        0.0,
        1.0,
        0.0,
        0.0,
    )
    .unwrap();

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                // Pro move: We're going to cover the screen completely. Trust us!
                load: DontCare,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();
    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
    let mut set_builder = PersistentDescriptorSet::start(layout.clone());

    set_builder
        .add_sampled_image(fb2d_texture, fb2d_sampler)
        .unwrap();

    let set = set_builder.build().unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let mut now_keys = [false; 255];
    let mut prev_keys = now_keys.clone();

    let mut now_lmouse = false;
    let mut prev_lmouse = false;
    let white = (255, 255, 255, 255);

    let w = 5_usize;
    let h = 10_usize;

    let mut player = Rect {
        x: WIDTH / 2 - w / 2,
        y: HEIGHT - h,
        width: w,
        height: h,
        color: white,
    };

    let mut dash = false;
    let mut dash_count = 0_u8;

    let mut vx = 0.0;
    let mut vy = 0.0;
    let mut ax = 0.0;
    let mut ay = 0.2;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::NewEvents(_) => {
                // Leave now_keys alone, but copy over all changed keys
                prev_keys.copy_from_slice(&now_keys);
                prev_lmouse = now_lmouse;
            }
            Event::WindowEvent {
                // Note this deeply nested pattern match
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            winit::event::KeyboardInput {
                                // Which serves to filter out only events we actually want
                                virtual_keycode: Some(keycode),
                                state,
                                ..
                            },
                        ..
                    },
                ..
            } => {
                // It also binds these handy variable names!
                match state {
                    winit::event::ElementState::Pressed => {
                        // VirtualKeycode is an enum with a defined representation
                        now_keys[keycode as usize] = true;
                    }
                    winit::event::ElementState::Released => {
                        now_keys[keycode as usize] = false;
                    }
                }
            }
            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        button: btn, state, ..
                    },
                ..
            } => {
                match btn {
                    winit::event::MouseButton::Left => {
                        match state {
                            winit::event::ElementState::Pressed => {
                                // VirtualKeycode is an enum with a defined representation
                                now_lmouse = true;
                            }
                            winit::event::ElementState::Released => {
                                now_lmouse = false;
                            }
                        }
                    }
                    _ => {}
                }
            }
            Event::MainEventsCleared => {
                {
                    // We need to synchronize here to send new data to the GPU.
                    // We can't send the new framebuffer until the previous frame is done being drawn.
                    // Dropping the future will block until it's done.
                    if let Some(mut fut) = previous_frame_end.take() {
                        fut.cleanup_finished();
                    }
                }

                // Mouse Events
                if now_lmouse && !prev_lmouse {}

                // Keyboard Events
                if now_keys[VirtualKeyCode::Escape as usize] {
                    *control_flow = ControlFlow::Exit;
                }
                if now_keys[VirtualKeyCode::LShift as usize]
                    || now_keys[VirtualKeyCode::RShift as usize]
                {
                } else {
                }

                if now_keys[VirtualKeyCode::Up as usize]
                    && !prev_keys[VirtualKeyCode::Up as usize]
                    && !dash
                {
                    vy = -5.0;
                }

                if now_keys[VirtualKeyCode::Down as usize]
                    && !prev_keys[VirtualKeyCode::Up as usize]
                    && !dash
                {
                    dash = true;
                    vx = 0.0;
                    vy = 0.0;
                    ax = 0.0;
                    ay = 0.0;
                }
                if now_keys[VirtualKeyCode::Left as usize] {
                    if dash {
                        dash = false;
                        ay = 0.2;
                        player.change_color_to(white);
                        vx = -(dash_count as f32 / 25.0);
                        dash_count = 0;
                    } else {
                        if vx > -2.0 {
                            ax = -0.2;
                        } else {
                            ax = 0.0
                        }
                    }
                } else if now_keys[VirtualKeyCode::Right as usize] {
                    if dash {
                        dash = false;
                        ay = 0.2;
                        player.change_color_to(white);
                        vx = dash_count as f32 / 25.0;
                        dash_count = 0;
                    } else {
                        if vx < 2.0 {
                            ax = 0.2;
                        } else {
                            ax = 0.0
                        }
                    }
                } else {
                    if vx > 0.09 {
                        ax = -0.1
                    } else if vx < -0.09 {
                        ax = 0.1
                    } else {
                        ax = 0.0
                    }
                }

                vx += ax;
                player.change_x_by(vx);

                vy += ay;
                player.change_y_by(vy);

                if dash {
                    player.change_color_to((255 - dash_count, 255, 255 - dash_count, 255));
                    if dash_count < 255 {
                        dash_count += 5;
                    }
                }

                clear(&mut fb2d, (128, 64, 64, 255));
                draw_filled_rect(&mut fb2d, player);

                // Now we can copy into our buffer.
                {
                    let writable_fb = &mut *fb2d_buffer.write().unwrap();
                    writable_fb.copy_from_slice(&fb2d);
                }

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate().dimensions(dimensions).build() {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut viewport,
                    );
                    recreate_swapchain = false;
                }
                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };
                if suboptimal {
                    recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    // Now copy that framebuffer buffer into the framebuffer image
                    .copy_buffer_to_image(fb2d_buffer.clone(), fb2d_image.clone())
                    .unwrap()
                    // And resume our regularly scheduled programming
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        std::iter::once(vulkano::format::ClearValue::None),
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        set.clone(),
                    )
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass()
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                let future = acquire_future
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new(image.clone()).unwrap();
            Framebuffer::start(render_pass.clone())
                .add(view)
                .unwrap()
                .build()
                .unwrap()
        })
        .collect::<Vec<_>>()
}
