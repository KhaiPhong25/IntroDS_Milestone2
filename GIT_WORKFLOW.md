# Git Workflow - PathForge Team

## 🌳 Branch Strategy

### Main Branch

- `main` - Branch chính của dự án, chứa code đã hoàn thành và stable

### Feature Branches

Mỗi member làm feature riêng trên branch riêng, tách từ `main`:

```
feature/ten-chuc-nang
```

**Ví dụ:**

- `feature/user-authentication`
- `feature/post-management`
- `feature/hotel-booking`
- `feature/travel-plan`

---

## 📋 Quy Trình Làm Việc

### 1️⃣ Bắt Đầu Feature Mới

```bash
# Đảm bảo đang ở branch main và cập nhật mới nhất
git checkout main
git pull origin main

# Tạo branch mới cho feature từ main
git checkout -b feature/ten-chuc-nang
```

### 2️⃣ Làm Việc Trên Feature

```bash
# Thường xuyên commit những thay đổi nhỏ
git add .
git commit -m "feat: mô tả ngắn gọn"

# Push lên remote để backup
git push origin feature/ten-chuc-nang
```

### 4️⃣ Hoàn Thành Feature

```bash
# Đảm bảo code đã test kỹ
# Push code lên remote
git push origin feature/ten-chuc-nang

# Tạo Pull Request trên GitHub
```

# Sau khi merge, cập nhật local
```bash
git checkout main
git pull origin main
git branch -d feature/ten-chuc-nang  # Xóa local branch
git push origin --delete feature/ten-chuc-nang  # Xóa remote branch (optional)

# Lưu ý: Chỉ xóa branch sau khi PR đã được merge
```

---

## 📝 Commit Message Convention

Sử dụng **Conventional Commits** để dễ theo dõi:

```
<type>: <description>

[optional body]
```

### Types

- `feat:` - Thêm feature mới
- `fix:` - Sửa bug
- `docs:` - Cập nhật documentation
- `style:` - Format code (không ảnh hưởng logic)
- `refactor:` - Refactor code
- `test:` - Thêm/sửa tests
- `chore:` - Cập nhật config, dependencies

### Ví Dụ

```bash
git commit -m "feat: add user login API endpoint"
git commit -m "fix: resolve JWT token expiration issue"
git commit -m "docs: update API documentation for auth routes"
```

---

## 🔄 Xử Lý Conflicts

Khi gặp conflict trong merge:

```bash
# 1. Mở file bị conflict, tìm dấu hiệu:
# <<<<<<< HEAD
# =======
# >>>>>>> develop

# 2. Sửa thủ công, giữ lại code đúng

# 3. Sau khi sửa xong
git add .
git commit -m "fix: resolve merge conflicts with develop"
```

---

## 🚨 Quy Tắc Quan Trọng

### ✅ NÊN

- Luôn tạo feature branch từ `main` mới nhất
- Commit thường xuyên với message rõ ràng
- Pull từ `main` trước khi bắt đầu làm việc
- Tạo PR và đợi review trước khi merge
- Test kỹ code trước khi tạo PR
- Sync với `main` thường xuyên để tránh conflict lớn

### ❌ KHÔNG NÊN

- **KHÔNG** commit trực tiếp vào `main` (luôn qua PR)
- **KHÔNG** force push (`git push -f`) vào branch đang có người làm chung
- **KHÔNG** commit file `.env`, `node_modules/`, `dist/`
- **KHÔNG** merge code chưa test
- **KHÔNG** để conflict quá lâu không xử lý

---

## 🛠️ Commands Hữu Ích

```bash
# Xem trạng thái hiện tại
git status

# Xem lịch sử commits
git log --oneline --graph

# Xem branch hiện tại
git branch

# Xem tất cả branches (kể cả remote)
git branch -a

# Hủy thay đổi chưa commit
git restore <file>

# Xem diff trước khi commit
git diff

# Stash changes tạm thời
git stash
git stash pop

# Xóa branch local
git branch -d feature/ten-branch
```

---
