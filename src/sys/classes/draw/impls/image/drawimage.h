#pragma once

#include <petscdraw.h>

typedef struct _n_PetscImage *PetscImage;
typedef struct _n_PetscImage {
  unsigned char *buffer;          /* raster buffer  */
  int            w, h;            /* width, height  */
  int            clip[4];         /* clip ranges    */
  unsigned char  palette[256][3]; /* colormap       */
} _n_PetscImage;

static inline void PetscImageSetClip(PetscImage img, int x, int y, int w, int h)
{
  img->clip[0] = PetscClipInterval(x, 0, img->w - 1); /* xmin   */
  img->clip[1] = PetscClipInterval(y, 0, img->h - 1); /* ymin   */
  img->clip[2] = PetscClipInterval(x + w, 0, img->w); /* xmax+1 */
  img->clip[3] = PetscClipInterval(y + h, 0, img->h); /* ymax+1 */
}

static inline void PetscImageClear(PetscImage img)
{
  int x, xs = img->clip[0], xe = img->clip[2];
  int y, ys = img->clip[1], ye = img->clip[3];
  for (y = ys; y < ye; y++)
    for (x = xs; x < xe; x++) img->buffer[y * img->w + x] = 0;
}

static inline void PetscImageDrawPixel(PetscImage img, int x, int y, int c)
{
  if (x < img->clip[0] || x >= img->clip[2]) return;
  if (y < img->clip[1] || y >= img->clip[3]) return;
  img->buffer[y * img->w + x] = (unsigned char)c;
}

static inline void PetscImageDrawLine(PetscImage img, int x_1, int y_1, int x_2, int y_2, int c)
{
  if (y_1 == y_2) {
    /* Horizontal line */
    if (x_2 - x_1 < 0) {
      int tmp = x_1;
      x_1     = x_2;
      x_2     = tmp;
    }
    while (x_1 <= x_2) PetscImageDrawPixel(img, x_1++, y_1, c);
  } else if (x_1 == x_2) {
    /* Vertical line */
    if (y_2 - y_1 < 0) {
      int tmp = y_1;
      y_1     = y_2;
      y_2     = tmp;
    }
    while (y_1 <= y_2) PetscImageDrawPixel(img, x_1, y_1++, c);
  } else {
    /* Bresenham's line drawing algorithm */
    int dx = PetscAbs(x_2 - x_1), sx = (x_2 - x_1) >= 0 ? +1 : -1;
    int dy = PetscAbs(y_2 - y_1), sy = (y_2 - y_1) >= 0 ? +1 : -1;
    int error = (dx > dy ? dx : -dy) / 2, err;
    while (1) {
      PetscImageDrawPixel(img, x_1, y_1, c);
      if (x_1 == x_2 && y_1 == y_2) break;
      err = error;
      if (err > -dx) {
        error -= dy;
        x_1 += sx;
      }
      if (err < +dy) {
        error += dx;
        y_1 += sy;
      }
    }
  }
}

static inline void PetscImageDrawRectangle(PetscImage img, int x, int y, int w, int h, int c)
{
  int xs = PetscMax(x, img->clip[0]), xe = PetscMin(x + w, img->clip[2]);
  int ys = PetscMax(y, img->clip[1]), ye = PetscMin(y + h, img->clip[3]);
  if (xs >= xe || ys >= ye) return;
  for (y = ys; y < ye; y++)
    for (x = xs; x < xe; x++) img->buffer[y * img->w + x] = (unsigned char)c;
}

static inline void PetscImageDrawEllipse(PetscImage img, int xc, int yc, int w, int h, int c)
{
  /* Bresenham's circle/ellipse drawing algorithm */
  int x, y, s, a2 = w * w, b2 = h * h;
  for (x = 0, y = h, s = 2 * b2 + a2 * (1 - 2 * h); b2 * x <= a2 * y; x++) {
    PetscImageDrawLine(img, xc + x, yc + y, xc - x, yc + y, c);
    PetscImageDrawLine(img, xc + x, yc - y, xc - x, yc - y, c);
    if (s >= 0) {
      s += 4 * a2 * (1 - y);
      y--;
    }
    s += b2 * ((4 * x) + 6);
  }
  for (x = w, y = 0, s = 2 * a2 + b2 * (1 - 2 * w); a2 * y <= b2 * x; y++) {
    PetscImageDrawLine(img, xc + x, yc + y, xc - x, yc + y, c);
    PetscImageDrawLine(img, xc + x, yc - y, xc - x, yc - y, c);
    if (s >= 0) {
      s += 4 * b2 * (1 - x);
      x--;
    }
    s += a2 * ((4 * y) + 6);
  }
}

static inline void PetscImageDrawTriangle(PetscImage img, int x_1, int y_1, int t_1, int x_2, int y_2, int t_2, int x_3, int y_3, int t_3)
{
  const int SHIFT_VAL = 6;
  const int xmin = img->clip[0], xmax = img->clip[2] - 1;
  const int ymin = img->clip[1], ymax = img->clip[3] - 1;
  float     rfrac, lfrac, one = 1;
  float     R_y2_y1, R_y3_y1, R_y3_y2;
  int       lc, rc = 0, lx, rx = 0, xx, y, c;
  int       rc_lc, rx_lx, t2_t1, x2_x1, t3_t1, x3_x1, t3_t2, x3_x2;

  /* Is triangle ever visible in image? */
  if (x_1 < xmin && x_2 < xmin && x_3 < xmin) return;
  if (y_1 < ymin && y_2 < ymin && y_3 < ymin) return;
  if (x_1 > xmax && x_2 > xmax && x_3 > xmax) return;
  if (y_1 > ymax && y_2 > ymax && y_3 > ymax) return;

  t_1 = t_1 << SHIFT_VAL;
  t_2 = t_2 << SHIFT_VAL;
  t_3 = t_3 << SHIFT_VAL;

  /* Sort the vertices */
#define SWAP(a, b) \
  do { \
    int _tmp; \
    _tmp = a; \
    a    = b; \
    b    = _tmp; \
  } while (0)
  if (y_1 > y_2) {
    SWAP(x_1, x_2);
    SWAP(y_1, y_2);
    SWAP(t_1, t_2);
  }
  if (y_1 > y_3) {
    SWAP(x_1, x_3);
    SWAP(y_1, y_3);
    SWAP(t_1, t_3);
  }
  if (y_2 > y_3) {
    SWAP(x_2, x_3);
    SWAP(y_2, y_3);
    SWAP(t_2, t_3);
  }
#undef SWAP

  /* This code is decidedly non-optimal;
   it is intended to be a start at an implementation */

  t2_t1   = t_2 - t_1;
  x2_x1   = x_2 - x_1;
  R_y2_y1 = (y_2 != y_1) ? one / (y_2 - y_1) : 0;
  R_y3_y1 = (y_3 != y_1) ? one / (y_3 - y_1) : 0;
  x3_x1   = x_3 - x_1;
  t3_t1   = t_3 - t_1;

  for (y = y_1; y <= y_2; y++) {
    /* Draw a line with the correct color from t1-t2 to t1-t3 */
    /* Left color is (y-y1)/(y2-y1) * (t2-t1) + t1 */
    lfrac = (y - y_1) * R_y2_y1;
    lc    = (int)(lfrac * (t2_t1) + t_1);
    lx    = (int)(lfrac * (x2_x1) + x_1);
    /* Right color is (y-y1)/(y3-y1) * (t3-t1) + t1 */
    rfrac = (y - y_1) * R_y3_y1;
    rc    = (int)(rfrac * (t3_t1) + t_1);
    rx    = (int)(rfrac * (x3_x1) + x_1);
    /* Draw the line */
    rc_lc = rc - lc;
    rx_lx = rx - lx;
    if (rx > lx) {
      for (xx = lx; xx <= rx; xx++) {
        c = (((xx - lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
        PetscImageDrawPixel(img, xx, y, c);
      }
    } else if (rx < lx) {
      for (xx = lx; xx >= rx; xx--) {
        c = (((xx - lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
        PetscImageDrawPixel(img, xx, y, c);
      }
    } else {
      c = lc >> SHIFT_VAL;
      PetscImageDrawPixel(img, lx, y, c);
    }
  }

  /* For simplicity,"move" t1 to the intersection of t1-t3 with the line y=y2.
     We take advantage of the previous iteration. */
  if (y_2 >= y_3) return;
  if (y_1 < y_2) {
    x_1   = rx;
    y_1   = y_2;
    t_1   = rc;
    x3_x1 = x_3 - x_1;
    t3_t1 = t_3 - t_1;
  }
  R_y3_y1 = (y_3 != y_1) ? one / (y_3 - y_1) : 0;
  R_y3_y2 = (y_3 != y_2) ? one / (y_3 - y_2) : 0;
  x3_x2   = x_3 - x_2;
  t3_t2   = t_3 - t_2;

  for (y = y_2; y <= y_3; y++) {
    /* Draw a line with the correct color from t2-t3 to t1-t3 */
    /* Left color is (y-y1)/(y2-y1) * (t2-t1) + t1 */
    lfrac = (y - y_2) * R_y3_y2;
    lc    = (int)(lfrac * (t3_t2) + t_2);
    lx    = (int)(lfrac * (x3_x2) + x_2);
    /* Right color is (y-y1)/(y3-y1) * (t3-t1) + t1 */
    rfrac = (y - y_1) * R_y3_y1;
    rc    = (int)(rfrac * (t3_t1) + t_1);
    rx    = (int)(rfrac * (x3_x1) + x_1);
    /* Draw the line */
    rc_lc = rc - lc;
    rx_lx = rx - lx;
    if (rx > lx) {
      for (xx = lx; xx <= rx; xx++) {
        c = (((xx - lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
        PetscImageDrawPixel(img, xx, y, c);
      }
    } else if (rx < lx) {
      for (xx = lx; xx >= rx; xx--) {
        c = (((xx - lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
        PetscImageDrawPixel(img, xx, y, c);
      }
    } else {
      c = lc >> SHIFT_VAL;
      PetscImageDrawPixel(img, lx, y, c);
    }
  }
}

#define PetscImageFontWidth  6
#define PetscImageFontHeight 10
static const unsigned char PetscImageFontBitmap[128 - 32][10] = {
  {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, /*   */
  {0x00, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00, 0x08, 0x00, 0x00}, /* ! */
  {0x00, 0x14, 0x14, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, /* " */
  {0x00, 0x14, 0x14, 0x3E, 0x14, 0x3E, 0x14, 0x14, 0x00, 0x00}, /* # */
  {0x00, 0x08, 0x1C, 0x28, 0x1C, 0x0A, 0x1C, 0x08, 0x00, 0x00}, /* $ */
  {0x00, 0x12, 0x2A, 0x14, 0x08, 0x14, 0x2A, 0x24, 0x00, 0x00}, /* % */
  {0x00, 0x10, 0x28, 0x28, 0x10, 0x2A, 0x24, 0x1A, 0x00, 0x00}, /* & */
  {0x00, 0x08, 0x08, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, /* ' */
  {0x00, 0x04, 0x08, 0x10, 0x10, 0x10, 0x08, 0x04, 0x00, 0x00}, /* ( */
  {0x00, 0x10, 0x08, 0x04, 0x04, 0x04, 0x08, 0x10, 0x00, 0x00}, /*) */
  {0x00, 0x00, 0x22, 0x14, 0x3E, 0x14, 0x22, 0x00, 0x00, 0x00}, /* * */
  {0x00, 0x00, 0x08, 0x08, 0x3E, 0x08, 0x08, 0x00, 0x00, 0x00}, /* + */
  {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x08, 0x10, 0x00}, /* , */
  {0x00, 0x00, 0x00, 0x00, 0x3E, 0x00, 0x00, 0x00, 0x00, 0x00}, /* - */
  {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x1C, 0x08, 0x00}, /* . */
  {0x00, 0x02, 0x02, 0x04, 0x08, 0x10, 0x20, 0x20, 0x00, 0x00}, /* / */
  {0x00, 0x08, 0x14, 0x22, 0x22, 0x22, 0x14, 0x08, 0x00, 0x00}, /* 0 */
  {0x00, 0x08, 0x18, 0x28, 0x08, 0x08, 0x08, 0x3E, 0x00, 0x00}, /* 1 */
  {0x00, 0x1C, 0x22, 0x02, 0x0C, 0x10, 0x20, 0x3E, 0x00, 0x00}, /* 2 */
  {0x00, 0x3E, 0x02, 0x04, 0x0C, 0x02, 0x22, 0x1C, 0x00, 0x00}, /* 3 */
  {0x00, 0x04, 0x0C, 0x14, 0x24, 0x3E, 0x04, 0x04, 0x00, 0x00}, /* 4 */
  {0x00, 0x3E, 0x20, 0x2C, 0x32, 0x02, 0x22, 0x1C, 0x00, 0x00}, /* 5 */
  {0x00, 0x0C, 0x10, 0x20, 0x2C, 0x32, 0x22, 0x1C, 0x00, 0x00}, /* 6 */
  {0x00, 0x3E, 0x02, 0x04, 0x04, 0x08, 0x10, 0x10, 0x00, 0x00}, /* 7 */
  {0x00, 0x1C, 0x22, 0x22, 0x1C, 0x22, 0x22, 0x1C, 0x00, 0x00}, /* 8 */
  {0x00, 0x1C, 0x22, 0x26, 0x1A, 0x02, 0x04, 0x18, 0x00, 0x00}, /* 9 */
  {0x00, 0x00, 0x08, 0x1C, 0x08, 0x00, 0x08, 0x1C, 0x08, 0x00}, /* : */
  {0x00, 0x00, 0x08, 0x1C, 0x08, 0x00, 0x0C, 0x08, 0x10, 0x00}, /* } */
  {0x00, 0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02, 0x00, 0x00}, /* < */
  {0x00, 0x00, 0x00, 0x3E, 0x00, 0x3E, 0x00, 0x00, 0x00, 0x00}, /* = */
  {0x00, 0x10, 0x08, 0x04, 0x02, 0x04, 0x08, 0x10, 0x00, 0x00}, /* > */
  {0x00, 0x1C, 0x22, 0x04, 0x08, 0x08, 0x00, 0x08, 0x00, 0x00}, /* ? */
  {0x00, 0x1C, 0x22, 0x26, 0x2A, 0x2C, 0x20, 0x1C, 0x00, 0x00}, /* @ */
  {0x00, 0x08, 0x14, 0x22, 0x22, 0x3E, 0x22, 0x22, 0x00, 0x00}, /* A */
  {0x00, 0x3C, 0x12, 0x12, 0x1C, 0x12, 0x12, 0x3C, 0x00, 0x00}, /* B */
  {0x00, 0x1C, 0x22, 0x20, 0x20, 0x20, 0x22, 0x1C, 0x00, 0x00}, /* C */
  {0x00, 0x3C, 0x12, 0x12, 0x12, 0x12, 0x12, 0x3C, 0x00, 0x00}, /* D */
  {0x00, 0x3E, 0x20, 0x20, 0x3C, 0x20, 0x20, 0x3E, 0x00, 0x00}, /* E */
  {0x00, 0x3E, 0x20, 0x20, 0x3C, 0x20, 0x20, 0x20, 0x00, 0x00}, /* F */
  {0x00, 0x1C, 0x22, 0x20, 0x20, 0x26, 0x22, 0x1C, 0x00, 0x00}, /* G */
  {0x00, 0x22, 0x22, 0x22, 0x3E, 0x22, 0x22, 0x22, 0x00, 0x00}, /* H */
  {0x00, 0x1C, 0x08, 0x08, 0x08, 0x08, 0x08, 0x1C, 0x00, 0x00}, /* I */
  {0x00, 0x0E, 0x04, 0x04, 0x04, 0x04, 0x24, 0x18, 0x00, 0x00}, /* J */
  {0x00, 0x22, 0x24, 0x28, 0x30, 0x28, 0x24, 0x22, 0x00, 0x00}, /* K */
  {0x00, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x3E, 0x00, 0x00}, /* L */
  {0x00, 0x22, 0x22, 0x36, 0x2A, 0x22, 0x22, 0x22, 0x00, 0x00}, /* M */
  {0x00, 0x22, 0x22, 0x32, 0x2A, 0x26, 0x22, 0x22, 0x00, 0x00}, /* N */
  {0x00, 0x1C, 0x22, 0x22, 0x22, 0x22, 0x22, 0x1C, 0x00, 0x00}, /* O */
  {0x00, 0x3C, 0x22, 0x22, 0x3C, 0x20, 0x20, 0x20, 0x00, 0x00}, /* P */
  {0x00, 0x1C, 0x22, 0x22, 0x22, 0x22, 0x2A, 0x1C, 0x02, 0x00}, /* Q */
  {0x00, 0x3C, 0x22, 0x22, 0x3C, 0x28, 0x24, 0x22, 0x00, 0x00}, /* R */
  {0x00, 0x1C, 0x22, 0x20, 0x1C, 0x02, 0x22, 0x1C, 0x00, 0x00}, /* S */
  {0x00, 0x3E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00}, /* T */
  {0x00, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x1C, 0x00, 0x00}, /* U */
  {0x00, 0x22, 0x22, 0x22, 0x14, 0x14, 0x14, 0x08, 0x00, 0x00}, /* V */
  {0x00, 0x22, 0x22, 0x22, 0x2A, 0x2A, 0x36, 0x22, 0x00, 0x00}, /* W */
  {0x00, 0x22, 0x22, 0x14, 0x08, 0x14, 0x22, 0x22, 0x00, 0x00}, /* X */
  {0x00, 0x22, 0x22, 0x14, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00}, /* Y */
  {0x00, 0x3E, 0x02, 0x04, 0x08, 0x10, 0x20, 0x3E, 0x00, 0x00}, /* Z */
  {0x00, 0x1C, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1C, 0x00, 0x00}, /* [ */
  {0x00, 0x20, 0x20, 0x10, 0x08, 0x04, 0x02, 0x02, 0x00, 0x00}, /* \ */
  {0x00, 0x1C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1C, 0x00, 0x00}, /* ] */
  {0x00, 0x08, 0x14, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, /* ^ */
  {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3E, 0x00}, /* _ */
  {0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, /* ` */
  {0x00, 0x00, 0x00, 0x1C, 0x02, 0x1E, 0x22, 0x1E, 0x00, 0x00}, /* a */
  {0x00, 0x20, 0x20, 0x2C, 0x32, 0x22, 0x32, 0x2C, 0x00, 0x00}, /* b */
  {0x00, 0x00, 0x00, 0x1C, 0x22, 0x20, 0x22, 0x1C, 0x00, 0x00}, /* c */
  {0x00, 0x02, 0x02, 0x1A, 0x26, 0x22, 0x26, 0x1A, 0x00, 0x00}, /* d */
  {0x00, 0x00, 0x00, 0x1C, 0x22, 0x3E, 0x20, 0x1C, 0x00, 0x00}, /* e */
  {0x00, 0x0C, 0x12, 0x10, 0x3C, 0x10, 0x10, 0x10, 0x00, 0x00}, /* f */
  {0x00, 0x00, 0x00, 0x1E, 0x22, 0x22, 0x1E, 0x02, 0x22, 0x1C}, /* g */
  {0x00, 0x20, 0x20, 0x2C, 0x32, 0x22, 0x22, 0x22, 0x00, 0x00}, /* h */
  {0x00, 0x08, 0x00, 0x18, 0x08, 0x08, 0x08, 0x1C, 0x00, 0x00}, /* i */
  {0x00, 0x02, 0x00, 0x06, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0C}, /* j */
  {0x00, 0x20, 0x20, 0x22, 0x24, 0x38, 0x24, 0x22, 0x00, 0x00}, /* k */
  {0x00, 0x18, 0x08, 0x08, 0x08, 0x08, 0x08, 0x1C, 0x00, 0x00}, /* l */
  {0x00, 0x00, 0x00, 0x34, 0x2A, 0x2A, 0x2A, 0x22, 0x00, 0x00}, /* m */
  {0x00, 0x00, 0x00, 0x2C, 0x32, 0x22, 0x22, 0x22, 0x00, 0x00}, /* n */
  {0x00, 0x00, 0x00, 0x1C, 0x22, 0x22, 0x22, 0x1C, 0x00, 0x00}, /* o */
  {0x00, 0x00, 0x00, 0x2C, 0x32, 0x22, 0x32, 0x2C, 0x20, 0x20}, /* p */
  {0x00, 0x00, 0x00, 0x1A, 0x26, 0x22, 0x26, 0x1A, 0x02, 0x02}, /* q */
  {0x00, 0x00, 0x00, 0x2C, 0x32, 0x20, 0x20, 0x20, 0x00, 0x00}, /* r */
  {0x00, 0x00, 0x00, 0x1C, 0x20, 0x1C, 0x02, 0x3C, 0x00, 0x00}, /* s */
  {0x00, 0x10, 0x10, 0x3C, 0x10, 0x10, 0x12, 0x0C, 0x00, 0x00}, /* t */
  {0x00, 0x00, 0x00, 0x22, 0x22, 0x22, 0x26, 0x1A, 0x00, 0x00}, /* u */
  {0x00, 0x00, 0x00, 0x22, 0x22, 0x14, 0x14, 0x08, 0x00, 0x00}, /* v */
  {0x00, 0x00, 0x00, 0x22, 0x22, 0x2A, 0x2A, 0x14, 0x00, 0x00}, /* w */
  {0x00, 0x00, 0x00, 0x22, 0x14, 0x08, 0x14, 0x22, 0x00, 0x00}, /* x */
  {0x00, 0x00, 0x00, 0x22, 0x22, 0x26, 0x1A, 0x02, 0x22, 0x1C}, /* y */
  {0x00, 0x00, 0x00, 0x3E, 0x04, 0x08, 0x10, 0x3E, 0x00, 0x00}, /* z */
  {0x00, 0x06, 0x08, 0x04, 0x18, 0x04, 0x08, 0x06, 0x00, 0x00}, /* { */
  {0x00, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00}, /* | */
  {0x00, 0x18, 0x04, 0x08, 0x06, 0x08, 0x04, 0x18, 0x00, 0x00}, /* } */
  {0x00, 0x12, 0x2A, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, /* ~ */
  {0x00, 0x2A, 0x00, 0x22, 0x00, 0x22, 0x00, 0x2A, 0x00, 0x00}, /* ASCII 127 */
};

static inline void PetscImageDrawText(PetscImage img, int x, int y, int c, const char text[])
{
  int i, j, k, tw = PetscImageFontWidth, th = PetscImageFontHeight;
  for (i = 0; i < th; i++) {
    for (k = 0; text[k]; k++) {
      int chr = PetscClipInterval(text[k], 32, 127);
      for (j = 0; j < tw; j++) {
        if (PetscImageFontBitmap[chr - 32][i] & (1 << (tw - 1 - j))) PetscImageDrawPixel(img, x + j + k * tw, y + i - th, c);
      }
    }
  }
}
