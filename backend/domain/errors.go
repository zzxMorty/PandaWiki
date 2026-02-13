package domain

import "errors"

var ErrModelNotConfigured = errors.New("model not configured")

var ErrPortHostAlreadyExists = errors.New("port and host already exists")

var ErrSyncCaddyConfigFailed = errors.New("failed to sync caddy config")

var ErrNodeParentIDInIDs = errors.New("node.parent_id in ids, can't delete")

var ErrPermissionDenied = errors.New("permission denied")

var ErrInternalServerError = errors.New("internal server error")

var ErrMaxNodeLimitReached = errors.New("max node limit reached")

var ErrMaxKnowledgeBaseLimitReached = errors.New("max knowledge base limit reached")

var ErrRAGServiceUnavailable = errors.New("rag service unavailable")
