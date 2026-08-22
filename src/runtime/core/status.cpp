#include "runtime/core/status.h"

namespace runtime{

    // ===========================
    // 1. Initializer
    // ===========================

    // Status::Status() = delete;
    Status::Status(StatusCode code)
        : code_(code)
    {}
    Status::Status(StatusCode code, std::string message)
        : code_(code),
          message_(std::move(message))
    {}

    // ===========================
    // 2. Query
    // ===========================

    bool Status::ok() const noexcept{
        return code_ == StatusCode::Ok;
    }

    StatusCode Status::code() const noexcept{
        return code_;
    }

    const std::string& Status::message() const noexcept{
        return message_;
    }

    // ==========================
    // 3. Factor
    // ==========================

    Status Status::OK(){
        return Status(StatusCode::Ok);
    }

    Status Status::Error(StatusCode code, std::string message){
        return Status(code, std::move(message));
    }
    
}